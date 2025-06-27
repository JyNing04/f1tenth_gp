import numpy as np
import casadi as cs
from f1tenth_ros2.params import f110
from f1tenth_ros2.models import dynamics, Ekinematic
from constraints import Boundary


class setupGPMPC:
	def __init__(self, horizon, Ts, Q, P, R, params, gpmodels, track_name, car=None,
				 track_constraints=False, gp_correction=True, input_acc=False):
		self.horizon = horizon
		self.Ts = Ts
		self.Q = Q
		self.P = P
		self.R = R
		self.params = params
		self.track_constraints = track_constraints
		self.gp_correction = gp_correction

		self.model = Ekinematic.Kinematic()
		self.track = track_name

		self.n_states = self.model.n_states
		self.n_inputs = self.model.n_inputs
		self.xref_dim = 2  # x, y reference only

		# GP models
		self.omega_gp = gpmodels['ω']
		self.beta_gp = gpmodels['β']
		self.x_scaler_std = gpmodels['xscaler'].scale_
		self.x_scaler_mean = gpmodels['xscaler'].mean_

		# CasADi variables
		self._define_problem()

	def _define_problem(self):
		nX, nU, N = self.n_states, self.n_inputs, self.horizon

		# Decision variables
		X = cs.SX.sym('X', nX, N + 1)
		U = cs.SX.sym('U', nU, N)

		# Parameters
		X0 = cs.SX.sym('X0', nX)
		XREF = cs.SX.sym('XREF', self.xref_dim, N + 1)
		UPREV = cs.SX.sym('UPREV', nU)

		if self.track_constraints:
			A_track = cs.SX.sym('A_track', 2 * N, 2)
			b_track = cs.SX.sym('b_track', 2 * N, 1)
			EPS = cs.SX.sym('EPS', 2, N)
		else:
			A_track = None
			b_track = None
			EPS = None

		# Constraints and cost
		constraints = []
		cost = 0

		# Initial constraint
		constraints.append(X[:, 0] - X0)

		# Loop over horizon
		for k in range(N):
			dynamics_rhs = self.model.casadi(X[:, k], U[:, k])

			if self.gp_correction:
				gp_input = (cs.vertcat(U[:, k], X[6, k], X[3:6, k]).T - self.x_scaler_mean) / self.x_scaler_std
				gp_error = cs.vertcat(
					0,
					0,
					0,
					self.omega_gp(gp_input)[0],
					self.beta_gp(gp_input)[0],
					0
				)
			else:
				gp_error = cs.DM.zeros(nX, 1)

			next_state = X[:, k] + self.Ts * dynamics_rhs + gp_error
			constraints.append(X[:, k + 1] - next_state)

		# Input and rate constraints
		for k in range(N):
			if k == 0:
				delta_steer = U[0, k] - UPREV[0]
			else:
				delta_steer = U[0, k] - U[0, k - 1]

			# Cost
			tracking_error = X[:self.xref_dim, k + 1] - XREF[:, k + 1]
			cost += tracking_error.T @ self.Q @ tracking_error
			cost += delta_steer.T * self.R[0, 0] * delta_steer
			cost += (U[1, k] * self.Ts) * self.R[1, 1] * (U[1, k] * self.Ts)

			# Input constraints
			constraints.append(U[0, k] - self.params['max_inputs'][0])
			constraints.append(-U[0, k] + self.params['min_inputs'][0])
			constraints.append(U[1, k] - self.params['max_rates'][1])
			constraints.append(-U[1, k] + self.params['min_rates'][1])

			# Track constraints
			if self.track_constraints:
				track_expr = A_track[2*k:2*k+2, :] @ X[:2, k + 1] - b_track[2*k:2*k+2, :]
				constraints.append(track_expr - EPS[:, k])
				cost += 1e6 * cs.dot(EPS[:, k], EPS[:, k])

		# Terminal cost
		terminal_error = X[:self.xref_dim, -1] - XREF[:, -1]
		cost += terminal_error.T @ self.P @ terminal_error

		# Collect all decision and parameter variables
		decision_vars = [cs.reshape(X, -1, 1), cs.reshape(U, -1, 1)]
		if self.track_constraints:
			decision_vars.append(cs.reshape(EPS, -1, 1))
		all_vars = cs.vertcat(*decision_vars)

		param_vars = [X0, cs.reshape(XREF, -1, 1), UPREV]
		if self.track_constraints:
			param_vars += [cs.reshape(A_track, -1, 1), cs.reshape(b_track, -1, 1)]
		all_params = cs.vertcat(*param_vars)

		# Build NLP problem
		nlp = {
			'x': all_vars,
			'p': all_params,
			'f': cost,
			'g': cs.vertcat(*constraints)
		}

		opts = {
			'ipopt': {'print_level': 0, 'max_iter': 100},
			'print_time': False
		}
		self.solver = cs.nlpsol('gpmpc_solver', 'ipopt', nlp, opts)

	def solve(self, x0, xref, uprev):
		N, nX, nU = self.horizon, self.n_states, self.n_inputs
		if self.track_constraints:
			A_all = np.zeros((2 * N, 2))
			b_all = np.zeros((2 * N, 1))
			for k in range(N):
				A_k, b_k = Boundary(xref[:, k + 1], self.track)
				A_all[2*k:2*k+2, :] = A_k
				b_all[2*k:2*k+2, :] = b_k
		else:
			A_all = np.zeros((0, 2))
			b_all = np.zeros((0, 1))

		p_list = [x0,
				  xref.T.reshape(-1),
				  uprev]
		if self.track_constraints:
			p_list += [A_all.reshape(-1), b_all.reshape(-1)]
		pvec = np.concatenate(p_list)

		# Solve
		num_eps = 2 * N if self.track_constraints else 0
		lbx = -np.inf * np.ones(nX * (N + 1) + nU * N + num_eps)
		ubx = np.inf * np.ones_like(lbx)

		lbg_eq = np.zeros(nX * (N + 1))
		lbg_ineq = -np.inf * np.ones(N * (4 + 2 * self.track_constraints))
		ubg_eq = np.zeros_like(lbg_eq)
		ubg_ineq = np.zeros_like(lbg_ineq)

		res = self.solver(
			p=pvec,
			lbx=lbx,
			ubx=ubx,
			lbg=np.concatenate([lbg_eq, lbg_ineq]),
			ubg=np.concatenate([ubg_eq, ubg_ineq])
		)

		# Extract result
		x_opt = res['x'].full().flatten()
		X_flat = x_opt[:nX*(N+1)].reshape((N+1, nX)).T
		U_flat = x_opt[nX*(N+1): nX*(N+1) + nU*N].reshape((N, nU)).T
		fval = res['f'].full().item()

		return U_flat, fval, X_flat