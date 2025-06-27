import numpy as np
import casadi as cs
import math
import os
import csv
from f1tenth_ros2.models import dynamics
from f1tenth_ros2.params import f110

class MPCCutils():
    def __init__(self, horizon, Q, P, R):
        vehicle_params  = f110.F110()

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

        self.max_inputs = vehicle_params['max_inputs']
        self.min_inputs = vehicle_params['min_inputs']
        self.max_rates  = vehicle_params['max_rates']
        self.min_rates  = vehicle_params['min_rates']

        

        self.n_states  = 7
        error_cor = False
        xref_size = 2
        
        # casadi variables
        x0    = cs.SX.sym('x0', self.n_states, 1)
        xref  = cs.SX.sym('xref', xref_size, horizon+1)
        uprev = cs.SX.sym('uprev', 2, 1)
        x     = cs.SX.sym('x', self.n_states, horizon+1)
        u     = cs.SX.sym('u', self.n_states, horizon)
        dxdtc = cs.SX.sym('dxdt', self.n_states, 1)
        
        # sum problem objectives and concatenate constraints
        cost_tracking = 0
        cost_actuation = 0
        cost_violation = 0

        if error_cor:
            pass
        else:
            cost_tracking += (x0[:xref_size,-1]-xref[:xref_size,-1]).T @ Q @ (x0[:xref_size,-1]-xref[:xref_size,-1])
