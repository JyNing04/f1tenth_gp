"""
Params for F1/10 race car
Author: Jingyun Ning
Email: jn2ne@virginia.edu
"""

def F110():

	mu = 1.0489 		    # Surface friction coefficient
	Csf = 4.718				# Cornering stiffness coefficient, front [1/rad]
	Csr = 5.4562			# Cornering stiffness coefficient, rear [1/rad]
	lf = 0.15875			# Distance from center of gravity to front axle [m]
	lr = 0.17145			# Distance from center of gravity to rear axle [m]
	hcog = 0.074			# Height of center of gravity [m]
	mass = 3.74 			# Total mass of the vehicle [kg]
	Iz = 0.04712 			# Moment of inertial of the entire vehicle about the z axis [kgm^2]
	min_steer = -0.4189 	# Minimum steering angle constraint [rad]
	max_steer = 0.4189 		# Maximum steering angle constraint [rad]
	max_steer_vel = 3.2 	# Maximum steering velocity constraint [rad/s]
	min_steer_vel = -3.2 	# Minimum steering velocity constraint [rad/s]
	min_v = -5.0			# Minimum longitudinal velocity [m/s]
	max_v = 20.				# Maximum longitudinal velocity [m/s]
	switch_v =7.319			# Switching velocity (velocity at which the acceleration is no longer able to create wheel spin) [m/s]
	max_acc = 9.51 			# max acceleration [m/s^2]
	min_acc = -13.26 		# max deceleration [m/s^2]
	width = 0.31			# width of the vehicle [m]
	length = 0.58			# length of the vehicle [m]
	g = 9.81  				# Gravitational acceleration [m/s^2]
	max_inputs = [max_acc, max_steer]
	min_inputs = [min_acc, min_steer]

	max_rates = [None, max_steer_vel]
	min_rates = [None, min_steer_vel]	

	params_dict = {
		'lf': lf,
		'lr': lr,
		'mass': mass,
		'Iz': Iz,
		'Csf': Csf,
		'Csr': Csr,
		'hcog': hcog,
		'mu': mu,
		'min_v': min_v,
		'max_v': max_v,
		'switch_v': switch_v,
		'max_acc': max_acc,
		'min_acc': min_acc,
		'max_steer': max_steer,
		'min_steer': min_steer,
		'max_steer_vel': max_steer_vel,
		'min_steer_vel': min_steer_vel,
		'width': width,
		'length': length,
		'max_inputs': max_inputs,
		'min_inputs': min_inputs,
		'max_rates': max_rates,
		'min_rates': min_rates,
		'g': g
		}
	return params_dict