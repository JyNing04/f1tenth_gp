'''
Visualize the raceline on track, with knowledge of vehicle width.
Used for testing feasibility of generated raceline
'''

from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import math
from scipy import signal

map       = "Sepang" # Shanghai, Sepang, YasMarina
path      = os.path.expanduser('~/dev_ws/src/f1tenth_ros2/Tracks/'+map)
race_file = map + "_raceline"
cent_file = map + "_centerline"
save      = False

def loading_trackfile(file, path):
    track    = []
    file_name = os.path.join(path, file + '.csv')
    with open(file_name, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter =',')
            next(csv_reader)
            for waypoint in csv_reader:
                track.append(waypoint)
            # Force elements in racetrack to be float
            race_track = np.array(track, dtype=np.float32)
            track_size = len(race_track)
            return race_track, track_size

centerline, c_size = loading_trackfile(cent_file, path)
raceline, r_size   = loading_trackfile(race_file, path)
raceline_re        = signal.resample(raceline, c_size)
car_width  = 0.34
car_delta  = 0.15

def construct_boundary(centerline, track_size):
    center_x   = centerline[:, 0]
    center_y   = centerline[:, 1]
    right2wall = centerline[:, 2]
    left2wall  = centerline[:, 3]
    theta_arr  = np.zeros(track_size)
    theta_arr[1:]  = np.arctan2(np.diff(center_y), np.diff(center_x)) + math.pi
    boundsR_xy  = np.zeros((track_size, 2))
    boundsL_xy  = np.zeros((track_size, 2))
    for idx in range(track_size):
        boundsR_xy[idx, 0] = center_x[idx] + right2wall[idx] * math.sin(theta_arr[idx])
        boundsR_xy[idx, 1] = center_y[idx] - right2wall[idx] * math.cos(theta_arr[idx]) 
        boundsL_xy[idx, 0] = center_x[idx] - left2wall[idx] * math.sin(theta_arr[idx])
        boundsL_xy[idx, 1] = center_y[idx] + left2wall[idx] * math.cos(theta_arr[idx]) 
    return boundsR_xy, boundsL_xy 

def vehicle_bounds(raceline, tracksize, car_width=None, car_delta=None):
    delta_dis = car_width / 2 + car_delta
    cog_pos_x = raceline[:, 0]
    cog_pos_y = raceline[:, 1]
    theta_arr  = np.zeros(tracksize)
    theta_arr[1:]  = np.arctan2(np.diff(cog_pos_y), np.diff(cog_pos_x)) + math.pi
    boundsR_xy  = np.zeros((tracksize, 2))
    boundsL_xy  = np.zeros((tracksize, 2))
    for idx in range(tracksize):
        boundsR_xy[idx, 0] = cog_pos_x[idx] + delta_dis * math.sin(theta_arr[idx])
        boundsR_xy[idx, 1] = cog_pos_y[idx] - delta_dis * math.cos(theta_arr[idx]) 
        boundsL_xy[idx, 0] = cog_pos_x[idx] - delta_dis * math.sin(theta_arr[idx])
        boundsL_xy[idx, 1] = cog_pos_y[idx] + delta_dis * math.cos(theta_arr[idx]) 
    return boundsR_xy, boundsL_xy 

def offset_determin(centerline, raceline, start_pt, length):
    """
    Determin the index offset between raceline and centerline at the same XY location
    """
    count  = 0
    thres_val = 1.7
    start_pos = np.asarray(start_pt)
    race_flag = False
    cen_flag  = False
    idx_rc = []
    idx_ce = []
    wpt_ls = []
    cen_ls = []

    for idx, ele in enumerate(raceline):
        cur_pos = np.asarray(ele[0:2])
        if np.linalg.norm(start_pos-cur_pos) < thres_val:
            race_flag = True 
        if race_flag and count < length:
            count += 1
            idx_rc.append(idx)
            wpt_ls.append(ele)

    for idx, ele in enumerate(centerline):
        cur_pos = np.asarray(ele[0:2])
        # print(np.linalg.norm(start_pos-cur_pos))
        if np.linalg.norm(start_pos-cur_pos) < thres_val:
            cen_flag = True 
        if cen_flag and count > 0:
            cen_ls.append(ele)
            idx_ce.append(idx)
            count -= 1

    offset = idx_ce[0] - idx_rc[0]
    # print("For XY position: [%.2f, %.2f]"%(start_pt[0], start_pt[1]))
    # print("Raceline index start from: %d, at location [%.2f, %.2f]"% (idx_rc[0], raceline[idx_rc[0]][0], raceline[idx_rc[0]][1]))
    # print("Centerline index start from: %d, at location [%.2f, %.2f]"% (idx_ce[0], centerline[idx_ce[0]][0], centerline[idx_ce[0]][1]))
    # print("Offset is: ", offset)
    return offset

def modify_raceline(centerline, raceline, offset, start_pt, length, delta=0.0):
    """
    Prevent Sharp turn or cut in too soon
    """
    thres_val = 1.0
    start_pos = np.asarray(start_pt)
    mod_flag  = False
    centl_len = len(centerline)
    for idx, ele in enumerate(raceline):
        count   = 0
        cur_pos = np.asarray(ele[0:2])
        if np.linalg.norm(start_pos-cur_pos) < thres_val:
            mod_flag = True 
        if mod_flag and count < length:
            count  += 1
            # Calc the offset between centerline
            offsetX = ele[0] - centerline[(idx+offset)%centl_len][0]
            offsetY = ele[1] - centerline[(idx+offset)%centl_len][1]
            # if offsetX > 0:
            ele[0] -= delta * offsetX
            ele[1] -= delta * offsetY
    return raceline

# Shanghai
# offset_1 = offset_determin(centerline, raceline_re, [42.5, -10], 15)
# raceline = modify_raceline(centerline, raceline_re, offset_1, [40, -10], 15, delta=0.3) # Fix turn 1
# offset_2 = offset_determin(centerline, raceline_re, [-49, 4], 15)
# raceline = modify_raceline(centerline, raceline, offset_2, [-49, 4], 20, delta=0.3) # Fix turn 2
# offset_3 = offset_determin(centerline, raceline_re, [-18.5, 66], 15)
# raceline = modify_raceline(centerline, raceline, offset_3, [-18.5, 66], 15, delta=0.3) # Fix turn 3

# Sepang
offset_1 = offset_determin(centerline, raceline_re, [20, -6], 15)
raceline = modify_raceline(centerline, raceline_re, offset_1, [20, -6], 15, delta=0.2) # Fix turn 1
offset_2 = offset_determin(centerline, raceline_re, [13.5, -12], 20)
raceline = modify_raceline(centerline, raceline_re, offset_2, [13.5, -12], 20, delta=0.2) # Fix turn 2
offset_3 = offset_determin(centerline, raceline_re, [-54, -4.2], 25)
raceline = modify_raceline(centerline, raceline_re, offset_3, [-54, -4.2], 25, delta=0.5) # Fix turn 3
offset_4 = offset_determin(centerline, raceline_re, [-41, -4], 25)
raceline = modify_raceline(centerline, raceline_re, offset_4, [-41, -4], 25, delta=0.2) # Fix turn 4

# YasMarina
# offset_1 = offset_determin(centerline, raceline_re, [26, 3], 20)
# raceline = modify_raceline(centerline, raceline_re, offset_1, [26, 3], 20, delta=0.2) # Fix turn 1
# offset_2 = offset_determin(centerline, raceline_re, [26.5, 19], 20)
# raceline = modify_raceline(centerline, raceline_re, offset_2, [26.5, 19], 20, delta=0.1) # Fix turn 2
# offset_3 = offset_determin(centerline, raceline_re, [12, 62], 25)
# raceline = modify_raceline(centerline, raceline_re, offset_3, [12, 62], 30, delta=0.6) # Fix turn 3
# offset_4 = offset_determin(centerline, raceline_re, [-18, -9], 25)
# raceline = modify_raceline(centerline, raceline_re, offset_4, [-18, -9], 35, delta=0.5) # Fix turn 4
# offset_5 = offset_determin(centerline, raceline_re, [0.3, 0], 5)
# raceline = modify_raceline(centerline, raceline_re, offset_5, [0.4, 0], 5, delta=0.3) # Fix starting straight

boundsL, boundsR   = construct_boundary(centerline, c_size)
vehicleL, vehicleR = vehicle_bounds(raceline_re, c_size, car_width  = 0.34, car_delta  = 0.15)


# Save raceline to .csv file
def save_file(file, path, raceline):
    file_name = os.path.join(path, file + '.csv')
    np.savetxt(file_name, raceline, delimiter=',')

if save:
    path = os.path.expanduser('~/dev_ws/src/f1tenth_ros2/f1tenth_ros2/data/tracks') 
    save_file(race_file+'_ED', path, raceline)
    print("Saving file: "+race_file+'_ED')

mpc_path = "/home/ning/dev_ws/src/f1tenth_ros2/f1tenth_ros2/data/mpc/raceline/f1tenth-DYN-MPC-Sepang_raceline_ED-plot.npz"
gpmpc_path = "/home/ning/dev_ws/src/f1tenth_ros2/f1tenth_ros2/data/mpc/raceline/f1tenth-EKIN-GPMPC-Sepang_raceline_ED.npz"
mpcdata = np.load(mpc_path)
gpmpcdata = np.load(gpmpc_path)

states   = mpcdata['states']
gpstates = gpmpcdata['states']
inputs   = mpcdata['inputs']
gpinputs = gpmpcdata['inputs']
mpcX     = states[:,0]
mpcY     = states[:,1]
speed    = states[:,3]
gpspeed  = gpstates[:,3]
mpc_step = states.shape[0]+2
gp_step  = gpstates.shape[0]+2
# print(np.average(speed))
# print(np.average(gpspeed))
mpcomega = states[:,5]
gpomega  = gpstates[:,5]
refXY    = mpcdata['refXY']
posXY    = mpcdata['posXY']
gmposXY  = gpmpcdata['posXY']
gmposXY  = np.delete(gmposXY, 1, 1)
mpcXY    = mpcdata['predXY']
gpmpcXY  = gpmpcdata['predXY']
# checkpt  = [[28.0, 6.9, -27.4, -22, 0.2, 30.3, 21.0, -18.2, -48.35, -30.65, 44.35], [2.3, -7.8, -13.76, -23.8, -15.5, -10.14, 21.8, 39.6, 1.38, -3.12, -3.3]]
gmposXY  = gmposXY[:, :gp_step+1]
posXY    = posXY[:, :mpc_step]

gmposXY  = gmposXY[:, ::2]
gp_step  = int(gp_step/2)
# print(mpc_step)
print(posXY.shape)

plt.figure(figsize=([12.6, 9.5]))
plt.plot(boundsL[:,0], boundsL[:,1], c='0.85', linewidth=1)
plt.plot(boundsR[:,0], boundsR[:,1], c='0.85', linewidth=1)
plt.plot(raceline[:,0], raceline[:,1], '--', c='0.85', linewidth=1)
plt.plot(posXY[0,-1], posXY[1,-1], 'g*',  markersize=15, label='Start/Finish point')
'''
# idx_fwd1 = 23
# idx_fwd2 = 15
# plt.plot(states[100:100+idx_fwd1,0], states[100:100+idx_fwd1,1], 'g--', label='DYNMPC (ground truth)', linewidth=4)
# plt.plot(states[201:201+idx_fwd1,0], states[201:201+idx_fwd1,1], 'g--', linewidth=4)
# plt.plot(states[274:274+idx_fwd1,0], states[274:274+idx_fwd1,1], 'g--', linewidth=4)
# plt.plot(states[427:427+idx_fwd1,0], states[427:427+idx_fwd1,1], 'g--', linewidth=4)
# plt.plot(states[612:612+idx_fwd1,0], states[612:612+idx_fwd1,1], 'g--', linewidth=4)
# plt.plot(states[705:705+idx_fwd1,0], states[705:705+idx_fwd1,1], 'g--', linewidth=4)
# plt.plot(states[755:755+idx_fwd1,0], states[755:755+idx_fwd1,1], 'g--', linewidth=4)
# plt.plot(states[872:872+idx_fwd1,0], states[872:872+idx_fwd1,1], 'g--', linewidth=4)
# plt.plot(states[1062:1062+idx_fwd1,0], states[1062:1062+idx_fwd1,1], 'g--', linewidth=4)
# plt.plot(states[1243:1243+idx_fwd1,0], states[1243:1243+idx_fwd1,1], 'g--', linewidth=4)
# plt.plot(states[1375:1375+idx_fwd1,0], states[1375:1375+idx_fwd1,1], 'g--', linewidth=4)
# plt.plot(gpstates[111:111+idx_fwd2,0], gpstates[111:111+idx_fwd2,1], 'r--', label='GPMPC (prediction)', linewidth=4)
# plt.plot(gpstates[175:175+idx_fwd2,0], gpstates[175:175+idx_fwd2,1], 'r--', linewidth=4)
# plt.plot(gpstates[235:235+idx_fwd2,0], gpstates[235:235+idx_fwd2,1], 'r--', linewidth=4)
# plt.plot(gpstates[316:316+idx_fwd2,0], gpstates[316:316+idx_fwd2,1], 'r--', linewidth=4)
# plt.plot(gpstates[428:428+idx_fwd2,0], gpstates[428:428+idx_fwd2,1], 'r--', linewidth=4)
# plt.plot(gpstates[495:495+idx_fwd2,0], gpstates[495:495+idx_fwd2,1], 'r--', linewidth=4)
# plt.plot(gpstates[523:523+idx_fwd2-3,0], gpstates[523:523+idx_fwd2-3,1], 'r--', linewidth=4)
# plt.plot(gpstates[586:586+idx_fwd2,0], gpstates[586:586+idx_fwd2,1], 'r--', linewidth=4)
# plt.plot(gpstates[699:699+idx_fwd2,0], gpstates[699:699+idx_fwd2,1], 'r--', linewidth=4)
# plt.plot(gpstates[805:805+idx_fwd2,0], gpstates[805:805+idx_fwd2,1], 'r--', linewidth=4)
# plt.plot(gpstates[890:890+idx_fwd2,0], gpstates[890:890+idx_fwd2,1], 'r--', linewidth=4)
# plt.plot(checkpt[0], checkpt[1], 'bo', markersize=15, label='Check point')

# plt.plot(mpcXY[0,0,:], mpcXY[0,1,:], 'g-', linewidth=4, label='ground truth')
# for ele in mpcXY[1:]:
#     plt.plot(ele[0,:], ele[1,:], 'g-', linewidth=4)

# plt.plot(gpmpcXY[0,0,:], gpmpcXY[0,1,:], 'r-', linewidth=4, label='predicted')
# for ele in gpmpcXY[1:]:
#     plt.plot(ele[0,:], ele[1,:], 'r-', linewidth=4)

# plt.plot(mpcXY[0,0,:], mpcXY[0,1,:], 'r-', linewidth=4, label='prediction')
# for ele in mpcXY[1:]:
#     plt.plot(ele[0,:], ele[1,:], 'r-', linewidth=4)
# plt.plot(states[:,0], states[:,1], 'g--', label='DYNMPC (ground truth)', linewidth=2)
# plt.plot(gpstates[:,0], gpstates[:,1], 'r--', linewidth=2, label='GPMPC (prediction)')
# plt.plot(posXY[0,0], posXY[1,0], 'g*',  markersize=15, label='Start point')
# plt.plot(gmposXY[0,1:-1], gmposXY[1,1:-1], 'ro', markersize=10, label='GPMPC check point')
# plt.plot(posXY[0,1:-2], posXY[1,1:-2], 'ko', markersize=10, label='MPC check point')
# plt.xlabel('x[m]', fontsize=20)
# plt.xticks(fontsize=15)
# plt.ylabel('y[m]', fontsize=20)
# plt.yticks(fontsize=15)
# plt.title("Track Position", fontsize=20)
# plt.legend(prop={'size': 16})
# plt.savefig('gpmpc_mpc.png', dpi=650)

# plt.figure(figsize=([12.6, 2.5]))
# accgp = signal.resample(gpinputs[:,0], 863)
# acc = signal.resample(inputs[:,0], 863)
# plt.plot(accgp, 'r-', label="GPMPC")
# plt.plot(acc, 'g-',label="DYNMPC")
# plt.xlabel('time step [$0.02 sec$]', fontsize=10)
# plt.xticks(fontsize=10)
# plt.ylabel('$a$ [$m/s^2$]', fontsize=20)
# plt.yticks(fontsize=10)
# plt.legend()
# plt.savefig('gpmpc_mpc_input1.png', dpi=650)

# plt.figure(figsize=([12.6, 2.5]))
# steer = signal.resample(inputs[:,1], 863)
# steergp = signal.resample(gpinputs[:,1], 863)
# plt.plot(steergp, 'r-', label="GPMPC", alpha=0.8)
# plt.plot(steer, 'g-', label="DYNMPC")
# plt.xlabel('time step [$0.02 sec$]', fontsize=10)
# plt.xticks(fontsize=10)
# plt.ylabel("$\Delta\delta$ [$rad/s$]", fontsize=20)
# plt.yticks(fontsize=10)
# plt.legend()
# plt.savefig('gpmpc_mpc_input2.png', dpi=650)


# plt.figure(figsize=([12.6, 9.5]))
# gpomega = signal.resample(gpomega, 863)
# mpcomega = signal.resample(mpcomega*1, 863)
# plt.plot(gpomega[10:], 'r-', label="GPMPC", alpha=0.8)
# plt.plot(mpcomega*1.5,'g-', label="DYNMPC")
# plt.xlabel('time step (0.02 sec)', fontsize=20)
# plt.xticks(fontsize=15)
# plt.ylabel("$\omega$", fontsize=20)
# plt.yticks(fontsize=15)
# plt.legend()
# plt.savefig('gpmpc_mpc_omega.png', dpi=650)
'''
# Dynamic plot
ax   = plt.gca()
GpL, = ax.plot(posXY[0,0], posXY[1,0], 'r', marker='o', markersize=5, alpha=0.5, label='GPMPC')
GpP, = ax.plot(posXY[0,0],  posXY[1,0], '-r', markersize=1, lw=0.8, label="GPMPC path")
DyL, = ax.plot(gmposXY[0,0], gmposXY[1,0], '-g', marker='o', markersize=5, lw=0.5, label="DYNMPC")
DyP, = ax.plot(gmposXY[0,0],  gmposXY[1,0], '-g', markersize=1, lw=0.8, label="DYNMPC path")
plt.xlabel('x $[m]$', fontsize=20)
plt.ylabel('y $[m]$', fontsize=20)
plt.title("Sepang Interational Circuit", fontsize=20)
plt.legend()
plt.ion()
plt.show()

# update plot
Ts = 0.04
for idx in range(mpc_step):
    if idx < gp_step:
        GpL.set_xdata(posXY[0,idx])
        GpL.set_ydata(posXY[1,idx])
        DyL.set_xdata(gmposXY[0,idx])
        DyL.set_ydata(gmposXY[1,idx])
        GpP.set_xdata(posXY[0, :idx])
        GpP.set_ydata(posXY[1, :idx])
        DyP.set_xdata(gmposXY[0, :idx+1])
        DyP.set_ydata(gmposXY[1, :idx+1])
    if idx < mpc_step and idx >= gp_step:
        GpL.set_xdata(posXY[0,idx])
        GpL.set_ydata(posXY[1,idx])
        GpP.set_xdata(posXY[0, :idx])
        GpP.set_ydata(posXY[1, :idx])
    plt.pause(Ts/100)
plt.ioff()

plt.show()