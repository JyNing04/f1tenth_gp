"""
Plot vehicle states saved in each .npz file
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import _pickle as pickle
from scipy import signal
from scipy.interpolate import interp1d
from matplotlib import cm, colors


#####################################################################
def plot_states(states, states_k):
    speed_p = 'Capped Speed' if ORIGINAL else 'Non-Capped Speed'
    # plots
    x   = states[:n_steps,0]
    y   = states[:n_steps,1]
    vel = states[:n_steps,3]
    vel_kin = states_k[:n_steps,3]
    # length  = 1400
    # time_ds = np.linspace(time[0], time[-1], length)
    # plot speed
    plt.figure()
    # Signal resample
    # vel_sig  = signal.resample(vel, length)
    plt.plot(time[:n_steps], vel)#, '--', label='V_dyn')
    # plt.plot(time[:n_steps], vel_kin, '-.', label='V_kin')
    plt.title(speed_p, fontsize=20)
    plt.xlabel('time [s]', fontsize=20)
    plt.ylabel('speed [m/s]', fontsize=20)
    plt.grid(True)
    plt.legend()

    # plot acceleration
    plt.figure()
    plt.plot(time[:n_steps], inputs[:n_steps, 0])
    plt.xlabel('time [s]')
    plt.ylabel('acceleration [m/s^2]')
    plt.grid(True)

    # plot steering velocity
    plt.figure()
    plt.plot(time[:n_steps], inputs[:n_steps, 1])
    plt.xlabel('time [s]')
    plt.ylabel('steering velocity [rad/s]')
    plt.grid(True)

    # # plot steering angle
    plt.figure()
    plt.plot(time[:n_steps], states[:n_steps,2])
    plt.xlabel('time [s]')
    plt.ylabel('steering [rad]')
    plt.grid(True)

    # # plot inertial heading
    plt.figure()
    plt.plot(time[:n_steps], states[:n_steps, 4])
    plt.xlabel('time [s]')
    plt.ylabel('orientation [rad]')
    plt.grid(True)

    # Heat map for velocity
    plt.figure()
    plt.hexbin(x,y,vel, gridsize=120, cmap='brg')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Velocity Heatmap')
    cb = plt.colorbar()
    cb.set_label('Velocity (m/s)')

    # Visualize the difference ω, β
    plt.figure()
    plt.plot(np.arange(len(vel)), states[:, 5]-states_k[:, 5], label='ω error')
    plt.ylabel('Yaw rate (ω) error [m/s]')
    plt.xlabel('time [s]')

    plt.figure()
    plt.plot(time[15:], states[15:, 6]-states_k[15:, 6], label='β error')
    plt.ylabel('Body slip angle (β) error [rad]')
    plt.xlabel('time [s]')


#####################################################################

from matplotlib import gridspec


def plot_true_predicted_variance(
        y_true, y_mu, y_std, id,
        x=None, xlabel=None, ylabel=None, 
        figsize=(8,6), plot_title=None):
    """ use only when both and mean predictions are known
    """

    y_true = y_true.flatten()
    y_mu = y_mu.flatten()
    y_std = y_std.flatten()
    
    l = y_true.shape[0]
    if x is None:
        x = range(l)

    plt.figure(figsize=figsize)
    plt.title(plot_title)
    gs = gridspec.GridSpec(3,1)
    
    # mean variance
    plt.subplot(gs[:-1,:])
    plt.plot(x, y_mu, '#990000', ls='-', lw=1.5, zorder=9, 
             label='predicted')
    plt.fill_between(x, (y_mu+2*y_std), (y_mu-2*y_std),
                     alpha=0.2, color='m', label='+-2$\sigma$')
    plt.plot(x, y_true, '#e68a00', ls='--', lw=1, zorder=9, 
             label='true')
    plt.legend(loc='upper right')
    plt.title('True vs Predicted ({})'.format(id), fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.legend(loc=0)

    # errors        
    plt.subplot(gs[2,:])
    plt.plot(x, np.abs(np.array(y_true).flatten()-y_mu), '#990000', 
             ls='-', lw=0.5, zorder=9)
    plt.fill_between(x, np.zeros([l,1]).flatten(), 2*y_std,
                     alpha=0.2, color='m')
    plt.title("Model Error and Predicted Variance", fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel('error ' + ylabel, fontsize=20)
    plt.tight_layout()

if __name__ == '__main__':
    ORIGINAL    = True
    FULL_DATA   = False
    CTYPE       = 'PP'
    test_mode   = False
    map_index   = 0
    map_list    = ['Sepang', 'Shanghai', 'YasMarina']
    map_name    = map_list[map_index]
    line_type   = 'raceline_ED' # raceline_ED or centerline
    save_path   = os.path.expanduser('~/dev_ws/src/f1tenth_ros2/f1tenth_ros2/data/gp_models/')
    data_path   = os.path.expanduser('~/dev_ws/src/f1tenth_ros2/f1tenth_ros2/data/')
    downsample  = ['halfsample', 'downsample']
    sample_idx  = 1 # 0:half or 1:one-third
    data_name   = ['f1tenth-DYN-{}-{}_{}', 'f1tenth-KIN-{}-{}_{}']
    if not FULL_DATA:
        data_name = ['f1tenth-'+downsample[sample_idx]+'-DYN-{}-{}_{}', 'f1tenth-'+downsample[sample_idx]+'-KIN-{}-{}_{}']
    if ORIGINAL:
        save_path += 'original/'
        data_path += 'original/'
    elif line_type == 'centerline':
        save_path += 'centerline/'
        data_path += 'centerline/'
    else:
        save_path += 'raceline/'
        data_path += 'raceline/'
    if test_mode:
        data_path += 'test/'
        data_name[0] += '_test'
        data_name[1] += '_test'
    # Data of dynamic model & kinematic model
    data_dyn  = np.load(data_path + data_name[0].format(CTYPE, map_name, line_type) + '.npz')
    data_kin  = np.load(data_path + data_name[1].format(CTYPE, map_name, line_type) + '.npz')
    states   = data_dyn['states']
    dstates  = data_dyn['dstates']
    states_k = data_kin['states']
    inputs   = data_dyn['inputs']
    time     = data_dyn['time']
    if test_mode:
        states   = data_dyn['states']
        dstates  = data_dyn['dstates']
        states_k = data_kin['states']
        inputs   = data_dyn['inputs']
        time     = data_dyn['time']
    n_steps  = time.shape[0]
    print(n_steps)
    plot_states(states, states_k)

    data_test = np.load("/home/ning/dev_ws/src/f1tenth_ros2/f1tenth_ros2/plots/plotstest_y.npz")
    testY = data_test['testY'].reshape(-1,)
    kinY  = states_k[3:,5]
    testY = signal.resample(testY, kinY.shape[0])
    gpomega = testY*2 + kinY
    plt.figure(figsize=([8.6, 5.9]))
    plt.plot(gpomega[2:], 'r-', label= 'GPMPC (prediction)', linewidth=2)
    plt.plot(states[:,5], 'g-', label= 'DYNMPC (ground truth)', linewidth=2)
    plt.xlabel('time step (0.02 sec)', fontsize=20)
    plt.xticks(fontsize=15)
    plt.ylabel('$\omega$', fontsize=20)
    plt.yticks(fontsize=15)
    # plt.title("GPMPC VS MPC", fontsize=20)
    plt.legend(prop={'size': 16})
    plt.rcParams['savefig.dpi'] = 600
    plt.show()