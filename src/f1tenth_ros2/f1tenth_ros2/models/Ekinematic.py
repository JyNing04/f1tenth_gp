"""	
Extended kinematic single track model.
"""

import numpy as np
from f1tenth_ros2.params import f110
import math
import casadi as cs

class Kinematic():
	def __init__(self):
		vehicle_params = f110.F110()
		self.g     = vehicle_params['g']
		self.lf    = vehicle_params['lf']
		self.lr    = vehicle_params['lr']
		self.mass  = vehicle_params['mass']
		self.Iz    = vehicle_params['Iz']
		self.Csf   = vehicle_params['Csf']
		self.Csr   = vehicle_params['Csr']
		self.hcog  = vehicle_params['hcog']
		self.mu    = vehicle_params['mu']
		self.min_v = vehicle_params['min_v']
		self.max_v = vehicle_params['max_v']
		self.s_v   = vehicle_params['switch_v']
		
		self.max_acc     = vehicle_params['max_acc']
		self.min_acc     = vehicle_params['min_acc']
		self.max_steer   = vehicle_params['max_steer']
		self.min_steer   = vehicle_params['min_steer']
		self.max_steer_v = vehicle_params['max_steer_vel']
		self.min_steer_v = vehicle_params['min_steer_vel']
		self.width       = vehicle_params['width']
		self.length      = vehicle_params['length']

		self.max_inputs  = vehicle_params['max_inputs']
		self.min_inputs  = vehicle_params['min_inputs']
		self.max_rates   = vehicle_params['max_rates']
		self.min_rates   = vehicle_params['min_rates']
		self.n_states    = 7
		self.n_inputs    = 2

	def sim_continuous(self, x0, x1, u, t):
		"""	simulates the nonlinear continuous model with given input vector
			by numerical integration using 6th order Runge Kutta method
			x0 is the initial state of size 1x7
			u is the input vector of size 2x1
			t is the time vector of size 1x2
		"""
		n_steps    = u.reshape(-1,1).shape[1] # 1
		ωβ_states  = 2 # ω & β
		x          = np.zeros([n_steps+1, self.n_states]) # size: 2x7
		x_ωβ       = np.zeros([n_steps+1, ωβ_states]) # size: 2x2
		dxdt       = np.zeros([n_steps+1, self.n_states]) # size: 2x7
		dxdt[0, :] = self.derivative_eqs(None, x0, [0, 0])
		x[0, :]    = x0
		x[1, :]    = x1
		for ids in range(1, n_steps+1):
			x_ωβ[ids, :] = self.odeintHeun2(x[ids-1, :], [t[ids-1],t[ids]], u)
			x[ids, -2:]  = x_ωβ[ids, :]
			dxdt[ids, :] = self.derivative_eqs(None, x[ids, :], u)
		
		return x, dxdt


	def derivative_eqs(self, t, x, u):
		"""	
		write dynamics as first order ODE: dxdt = f(x(t))
		x is a 7x1 vector: [x, y, 𝛿, v, phi, ω, β]^T
		u is a 2x1 vector: [acc, Δ𝛿]^T
		"""
		acc     = u[0] 
		steer_v = u[1]
		𝛿       = x[2] 
		v       = x[3]
		phi     = x[4]
		ω       = x[5] 
		β       = x[6] 

		dx    = np.zeros(7)
		dx[0] = v * np.cos(phi + β)
		dx[1] = v * np.sin(phi + β)
		dx[2] = steer_v
		dx[3] = acc
		dx[4] = ω 
		# dω/dt, dβ/dt are Different from DST model
		dx[5] = (steer_v * v + acc * 𝛿) * self.lr / (self.lf+self.lr)
		dx[6] = (steer_v * v + acc * 𝛿) * (self.lf+self.lr)
		return dx

	def odeintHeun2(self, y0, t, u):
		'''
		x: [x, y, 𝛿, v, phi, ω, β]^T
		u: [acc, Δ𝛿]^T
		'''
		A      = 2/3 # dependence of the stages on derivatives
		B      = np.asarray([1/4, 3/4]) # quadrature  weights
		C      = 2/3 # nodes weights within the step
		y_next = np.zeros([len(t)-1, 2])
		fun    = self.derivative_eqs
		for i in range(len(t)-1):
			h      = t[i+1] - t[i]
			K      = np.zeros((len(B), len(B)))
			K[0]   = h * fun(0, y0, u)[-2:]
			y_ωβ   = y0[-2:] + A*K[0]
			K[1]   = h * fun(C*h, np.concatenate((y0[0:-2],y_ωβ)), u)[-2:]

			y_next[i, :] = y0[-2:] + B@K
			y0[-2:]      = y0[-2:] + B@K
		return y_next[-1,:]
	
	def odeintRK6(self, y0, h, u):
		A = np.asarray([[1/3],
						[0, 2/3],
						[1/12, 1/3, -1/12],
						[25/48, -55/24, 35/48, 15/8],
						[3/20, -11/20, -1/8, 1/2, 1/10],
						[-261/260, 33/13, 43/156, -118/39, 32/195, 80/39]]) # dependence of the stages on derivatives
		B      = np.asarray([13/200, 0, 11/40, 11/40, 4/25, 4/25, 13/200]) # quadrature  weights
		C      = np.asarray([1/3, 2/3, 1/3, 5/6, 1/6, 1]) # nodes weights within the step
		fun    = self.derivative_eqs
		y_next = np.zeros([1, len(y0)])
		K      = np.zeros((len(B), len(B)))
		K[0]   = h * fun(0, y0, u)
		K[1]   = h * fun(C[0]*h, y0+A[0][0]*K[0], u)
		K[2]   = h * fun(C[1]*h, y0+A[1][0]*K[0]+A[1][1]*K[1], u)
		K[3]   = h * fun(C[2]*h, y0+A[2][0]*K[0]+A[2][1]*K[1]+A[2][2]*K[2], u)
		K[4]   = h * fun(C[3]*h, y0+A[3][0]*K[0]+A[3][1]*K[1]+A[3][2]*K[2]+A[3][3]*K[3], u)
		K[5]   = h * fun(C[4]*h, y0+A[4][0]*K[0]+A[4][1]*K[1]+A[4][2]*K[2]+A[4][3]*K[3]+A[4][4]*K[4], u)
		K[6]   = h * fun(C[5]*h, y0+A[5][0]*K[0]+A[5][1]*K[1]+A[5][2]*K[2]+A[5][3]*K[3]+A[5][4]*K[4]+A[5][5]*K[5], u)
		
		y_next = y0 + B@K
		print(y_next[-2:] )
		return y_next[-2:]   

	def casadi(self, x, u, dxdt):
		"""	write dynamics as first order ODE: dxdt = f(x(t))
			x is a 6x1 vector: [x, y, 𝛿, v, phi, ω, β]^T
			u is a 2x1 vector: [acc, Δ𝛿]^T
			dxdt is a casadi.SX variable
		"""
		steer_v = u[1]
		acc = u[0]
		𝛿   = x[2] 
		v   = x[3]		
		phi = x[4]
		ω   = x[5] 
		β   = x[6] 

		v = cs.if_else(v < self.min_v, self.min_v, v)
		ω = cs.if_else(v < self.min_v, 0, ω)
		𝛿 = cs.if_else(v < self.min_v, 0, 𝛿)
		
		dxdt[0] = v * cs.cos(phi + β)
		dxdt[1] = v * cs.sin(phi + β)
		dxdt[2] = steer_v
		dxdt[3] = acc
		dxdt[4] = ω 
		dxdt[5] = (steer_v * v + acc * 𝛿) * self.lr / (self.lf+self.lr)
		dxdt[6] = (steer_v * v + acc * 𝛿) * (self.lf+self.lr)
		return dxdt
