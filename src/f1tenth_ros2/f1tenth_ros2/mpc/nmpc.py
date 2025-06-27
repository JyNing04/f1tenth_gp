import numpy as np
import casadi as cs
import sys
import os
path = os.path.expanduser('~/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/mpc')
sys.path.insert(1, path)
from constraints import Boundary


class setupMPC:
	def __init__(self, horizon, Ts, Q, P, R, S, vehicle_params, model, track_width, racetrack,
				 projection_constraint=False, track_constraint=False):
		self.horizon = horizon
		self.Ts = Ts
		self.Q = Q
		self.P = P
		self.R = R
		self.S = S
		self.params = vehicle_params
		self.model = model
		self.track_width = track_width
		self.track = racetrack
		self.projection_constraint = projection_constraint
		self.track_constraint = track_constraint

		self.boundary_limit = (self.track_width - vehicle_params['width']) / 2
		self._define_problem()

	def _define_problem(self):
		N = self.horizon
		nX = self.model.n_states
		nU = self.model.n_inputs

		# Decision variables and parameters
		X = cs.SX.sym("X", nX, N + 1)
		U = cs.SX.sym("U", nU, N)
		X0 = cs.SX.sym("X0", nX)
		XREF = cs.SX.sym("XREF", 2, N + 1)
		VYAWREF = cs.SX.sym("VYAWREF", 2, N + 1)
		UPREV = cs.SX.sym("UPREV", nU)
		DIST_PROJ = cs.SX.sym("DIST_PROJ", 1)

		if self.track_constraint:
			A_track = cs.SX.sym("A_track", 2 * N, 2)
			b_track = cs.SX.sym("b_track", 2 * N, 1)
			EPS = cs.SX.sym("EPS", 2, N)
		else:
			A_track, b_track, EPS = None, None, None

		constraints = []
		cost_tracking = 0
		cost_velocity = 0
		cost_actuation = 0
		cost_violation = 0

		constraints.append(X[:, 0] - X0)

		# set constraints
		for k in range(N):
			x_next = X[:, k] + self.Ts * self.model.casadi(X[:, k], U[:, k])
			constraints.append(X[:, k + 1] - x_next)

		for k in range(N):
			# Reference tracking
			tracking_err = X[:2, k + 1] - XREF[:, k + 1]
			cost_tracking += tracking_err.T @ self.Q @ tracking_err
			vel_yaw_err = X[3:5, k + 1] - VYAWREF[:, k + 1]
			cost_velocity += vel_yaw_err.T @ self.S @ vel_yaw_err

			if k == 0:
				delta_u = U[:, k] - UPREV
			else:
				delta_u = U[:, k] - U[:, k - 1]

			actuation_cost = U[:, k].T @ self.R @ U[:, k]
			cost_actuation += actuation_cost

			constraints += [
				U[0, k] - self.params['max_inputs'][0],
				-U[0, k] + self.params['min_inputs'][0],
				X[2, k] - self.params['max_inputs'][1],
				-X[2, k] + self.params['min_inputs'][1],
				delta_u[1] - self.params['max_rates'][1],
				-delta_u[1] + self.params['min_rates'][1]
			]

			# Track constraints
			if self.track_constraint:
				track_expr = A_track[2 * k:2 * k + 2, :] @ X[:2, k + 1] - b_track[2 * k:2 * k + 2, :]
				constraints.append(track_expr - EPS[:, k])
				cost_violation += 1e6 * cs.dot(EPS[:, k], EPS[:, k])

			# Projection cost
			if self.projection_constraint:
				cost_violation += DIST_PROJ * 2.5

		# Terminal cost
		terminal_err = X[:2, -1] - XREF[:, -1]
		cost_tracking += terminal_err.T @ self.P @ terminal_err

		terminal_vel_yaw_err = X[3:5, -1] - VYAWREF[:, -1]
		cost_velocity += terminal_vel_yaw_err.T @ self.P @ terminal_vel_yaw_err

		total_cost = cost_tracking + cost_velocity + cost_actuation + cost_violation

		# Collect decision variables
		vars_list = [cs.reshape(X, -1, 1), cs.reshape(U, -1, 1)]
		if self.track_constraint:
			vars_list.append(cs.reshape(EPS, -1, 1))
		decision_vars = cs.vertcat(*vars_list)

		# Collect parameter variables
		param_list = [
			X0,
			cs.reshape(XREF, -1, 1),
			UPREV,
			cs.reshape(VYAWREF, -1, 1),
			DIST_PROJ
		]
		if self.track_constraint:
			param_list += [cs.reshape(A_track, -1, 1), cs.reshape(b_track, -1, 1)]
		parameters = cs.vertcat(*param_list)

		# NLP problem
		nlp = {
			"x": decision_vars,
			"p": parameters,
			"f": total_cost,
			"g": cs.vertcat(*constraints)
		}

		opts = {
			"ipopt": {"print_level": 0, "max_iter": 100},
			"print_time": False
		}
		self.solver = cs.nlpsol("mpc_solver", "ipopt", nlp, opts)

	def solve(self, x0, xref, uprev, vyaw_ref, dist_proj):
		N = self.horizon
		nX = self.model.n_states
		nU = self.model.n_inputs

		# Build track constraints
		if self.track_constraint:
			A_all = np.zeros((2 * N, 2))
			b_all = np.zeros((2 * N, 1))
			for k in range(N):
				A_k, b_k = Boundary(xref[:, k + 1], self.track_width, self.track)
				A_all[2*k:2*k+2, :] = A_k
				b_all[2*k:2*k+2, :] = b_k
		else:
			A_all = np.zeros((0, 2))
			b_all = np.zeros((0, 1))

		# Flatten parameter vector
		param_list = [
			x0,
			xref.T.flatten(),
			uprev,
			vyaw_ref.T.flatten(),
			np.array([dist_proj])
		]
		if self.track_constraint:
			param_list += [A_all.flatten(), b_all.flatten()]
		pvec = np.concatenate(param_list)

		num_eps = 2 * N if self.track_constraint else 0
		nx = nX * (N + 1)
		nu = nU * N

		lbx = -np.inf * np.ones(nx + nu + num_eps)
		ubx = np.inf * np.ones_like(lbx)

		lbg_eq = np.zeros(nX * (N + 1))
		lbg_ineq = -np.inf * np.ones(N * (6 + 2 * self.track_constraint))
		ubg_eq = np.zeros_like(lbg_eq)
		ubg_ineq = np.zeros_like(lbg_ineq)

		res = self.solver(
			p=pvec,
			lbx=lbx,
			ubx=ubx,
			lbg=np.concatenate([lbg_eq, lbg_ineq]),
			ubg=np.concatenate([ubg_eq, ubg_ineq])
		)

		# Extract results
		x_opt = res["x"].full().flatten()
		X_res = x_opt[:nx].reshape((N + 1, nX)).T
		U_res = x_opt[nx: nx + nu].reshape((N, nU)).T
		fval = res["f"].full().item()

		return U_res, fval, X_res
