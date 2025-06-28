'''
Visualize the raceline on track, with knowledge of vehicle width.
Used for testing feasibility of generated raceline
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import math
from scipy import signal

map       = "Sepang" # Shanghai, Sepang, YasMarina
path      = os.path.expanduser('~/f1tenth_gp/src/f1tenth_ros2/Tracks/'+map)
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
    print("For XY position: [%.2f, %.2f]"%(start_pt[0], start_pt[1]))
    print("Raceline index start from: %d, at location [%.2f, %.2f]"% (idx_rc[0], raceline[idx_rc[0]][0], raceline[idx_rc[0]][1]))
    print("Centerline index start from: %d, at location [%.2f, %.2f]"% (idx_ce[0], centerline[idx_ce[0]][0], centerline[idx_ce[0]][1]))
    print("Offset is: ", offset)
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
    path = os.path.expanduser('~/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/tracks') 
    save_file(race_file+'_ED', path, raceline)
    print("Saving file: "+race_file+'_ED')

plt.figure()
plt.plot(boundsL[:,0], boundsL[:,1], 'k-')
plt.plot(boundsR[:,0], boundsR[:,1], 'k-')
plt.plot(centerline[:,0], centerline[:,1], 'r--', label='center line')
plt.plot(raceline[:,0], raceline[:,1],'g-', label="race line")
plt.plot(vehicleL[:,0], vehicleL[:,1], 'g.-', label='car left bound')
plt.plot(vehicleR[:,0], vehicleR[:,1], 'g.-', label='car right bound')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Track Map")
plt.legend()
plt.savefig('file.jpg', dpi=700)
plt.show()


