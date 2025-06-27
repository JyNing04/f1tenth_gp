"""	
Dynamic single track model.
"""

import numpy as np
import math
from f1tenth_ros2.params import f110
import casadi as cs


class Dynamic():
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

	def calc_forces(self, x, u, casadi=False):
		acc = u[0] 
		𝛿   = x[2] 
		β   = x[6] 
		ω   = x[5] 
		v   = x[3]
		if casadi:
			Fyf = 2 * self.mu * self.Csf * (self.mass * self.g * self.lr - self.mass * acc * self.hcog)*(𝛿 - β - self.lf*ω/(v+0.01))/(self.lf + self.lr)
			Fyr = 2 * self.mu * self.Csr * (self.mass * self.g * self.lf + self.mass * acc * self.hcog)*(-β + self.lr*ω/(v+0.01))/(self.lf + self.lr)
		else:
			Fyf = 2 * self.mu * self.Csf * (self.mass * self.g * self.lr - self.mass * acc * self.hcog)*(𝛿 - β - self.lf*ω/v)/(self.lf + self.lr)
			Fyr = 2 * self.mu * self.Csr * (self.mass * self.g * self.lf + self.mass * acc * self.hcog)*(-β + self.lr*ω/v)/(self.lf + self.lr)
			# Set limits of lateral forces
			max_Fyf = 70.
			max_Fyr = 20.
			Fyf     = Fyf if Fyf <= max_Fyf else math.copysign(max_Fyf, Fyf)
			Fyr     = Fyr if Fyr <= max_Fyr else math.copysign(max_Fyr, Fyr)
			# print("Fyf: {}, Fyr: {}, Fxr: {}".format(Fyf, Fyr, Fxr))
		Fxr = self.mass * acc
		return Fyf, Fyr, Fxr

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

		Fyf, Fyr, Fxr = self.calc_forces(x, u)
		dx    = np.zeros(7)
		dx[0] = v * np.cos(phi + β)
		dx[1] = v * np.sin(phi + β)
		dx[2] = steer_v
		dx[3] = acc
		dx[4] = ω 
		dx[5] = 1/self.Iz * (self.lf * Fyf * np.cos(𝛿) - self.lr * Fyr)
		dx[6] = 1/(self.mass * v) * (Fyf + Fyr) - ω 
		return dx

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
		return y_next[-2:]

	def odeintHeun2(self, y0, h, u):
		'''
		x: [x, y, 𝛿, v, phi, ω, β]^T
		u: [acc, Δ𝛿]^T
		'''
		A      = 2/3 # dependence of the stages on derivatives
		B      = np.asarray([1/4, 3/4]) # quadrature  weights
		C      = 2/3 # nodes weights within the step
		fun    = self.derivative_eqs
		y_next = np.zeros([1, 2])
		K      = np.zeros((len(B), len(B)))
		K[0]   = h * fun(0, y0, u)[-2:]
		y_ωβ   = y0[-2:] + A*K[0]
		K[1]   = h * fun(C*h, np.concatenate((y0[0:-2],y_ωβ)), u)[-2:]

		y_next = y0[-2:] + B@K

		return y_next

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
		

		Fyf, Fyr, Fxr = self.calc_forces(x, u, casadi=True)
		
		dxdt[0] = v * cs.cos(phi + β)
		dxdt[1] = v * cs.sin(phi + β)
		dxdt[2] = steer_v
		dxdt[3] = acc
		dxdt[4] = ω 
		dxdt[5] = 1/self.Iz * (self.lf * Fyf * cs.cos(𝛿) - self.lr * Fyr)
		dxdt[6] = 1/(self.mass * (v+0.01)) * (Fyf + Fyr) - ω
		return dxdt