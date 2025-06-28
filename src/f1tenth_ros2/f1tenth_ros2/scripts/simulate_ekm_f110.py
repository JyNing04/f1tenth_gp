"""	
Generate data by simulating extended kinematic model.
"""

import numpy as np
from f1tenth_ros2.models import Ekinematic
from f1tenth_ros2.params import f110
import matplotlib.pyplot as plt

# Settings

SAVE_RESULTS  = True
ORIGINAL	  = True 
test_mode     = True
CTYPE         = 'PP'
track_id	  = 1
track_name_ls = ['Sepang', 'Shanghai', 'YasMarina']
race_type     = 'centerline' # centerline or raceline
TRACK_NAME    = track_name_ls[track_id] + '_centerline' if 'centerline' in race_type else track_name_ls[track_id] + '_raceline_ED'
SAMPLING_TIME  = 0.021
dyn_fname = 'f1tenth-DYN-{}-{}.npz'
kin_fname = 'f1tenth-KIN-{}-{}.npz'
data_path = '/home/ning/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/centerline/' if 'center' in race_type else '/home/ning/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/raceline'
# Load vehicle params

params = f110.F110()
model  = Ekinematic.Kinematic()

# Load inputs used to simulate Dynamic model
if ORIGINAL:
	data_path = '/home/ning/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/original/'
	
if test_mode:
	data_path += 'test/'
	dyn_fname = 'f1tenth-DYN-{}-{}_test.npz'
	kin_fname = 'f1tenth-KIN-{}-{}_test.npz'

data       = np.load(data_path+dyn_fname.format(CTYPE, TRACK_NAME))
N_SAMPLES  = data['inputs'].shape[0]

time       = data['time']
states_dyn = data['states']
inputs_dyn = data['inputs']
print(time.shape[0])
# Open-loop simulation

n_states   = states_dyn.shape[1]
n_inputs   = inputs_dyn.shape[1]
states_kin = states_dyn

for idn in range(N_SAMPLES-1):
	x_next, dxdt_next    = model.sim_continuous(states_dyn[idn, :], states_dyn[idn+1, :], inputs_dyn[idn, :], [0, SAMPLING_TIME])
	states_kin[idn+1, :] = x_next[-1, :]

if SAVE_RESULTS:
	print('Save data to '+data_path+kin_fname.format(CTYPE, TRACK_NAME))
	np.savez(
		data_path+kin_fname.format(CTYPE, TRACK_NAME),
		time=time,
		states=states_kin,
		inputs=inputs_dyn,
		)

# Visualize the difference ω, β

plt.figure()
plt.plot(time[10:], states_kin[10:, 5], label='ω kinematic')
plt.plot(time[10:], data['states'][10:, 5], '--', label='ω dynamic')

plt.ylabel('Yaw rate (ω) [m/s]')
plt.xlabel('time [s]')
plt.legend()

plt.figure()
plt.plot(time[10:], states_kin[10:, 6], label='β kinematic')
plt.plot(time[10:], data['states'][10:, 6], '--', label='β dynamic')
plt.ylabel('Body slip angle (β) [rad]')
plt.xlabel('time [s]')
plt.legend()

plt.show()
