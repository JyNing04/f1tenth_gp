"""
Extract results from csv data file
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

# Load csv file
file = np.genfromtxt("/home/ning/dev_ws/src/f1tenth_ros2/f1tenth_ros2/data/gpresults.csv", delimiter=',', dtype=str)
col  = 12
##########################
# Calculate total time
##########################
def calc_sum(data):
    total_time = []
    sumT       = 0
    for ele in data:
        if ele == 'n/a' or ele == '':
            continue
        total_time.append(float(ele))
        sumT += float(ele)
    return sumT, len(total_time)

def scen_value(file, idx_size, scen_idx, dtype, kernel=None, kernel_id=None):
    scenValue = np.zeros(idx_size)
    for i in range(idx_size):
        if kernel:
            sumT = float(file[scen_idx[i],1:][kernel_id])
            num  = 1
        else:
            sumT, num = calc_sum(file[scen_idx[i],1:])
        if dtype == 'total':
            scenValue[i] = sumT
        elif dtype == 'mean':
            scenValue[i] = sumT / num
    return scenValue, num

# Total time of each scenarios 
def loop_track(map_name, dtype, start_idx, scen_name, col_diff, row_diff, kernel=None, kernel_id=None):
    map_value = {}
    for i in range(map_size):
        map = map_name[i]
        setime_dic = {}
        count = 0
        startId = start_idx[i]
        for j in range(len(scen_name)):
            count   += 1
            start    = startId+(count-1)*row_diff
            scen_idx = [start, start+col_diff, start+col_diff*2]
            if kernel:
                scenValue, num = scen_value(file, map_size, scen_idx, dtype, kernel, kernel_id)
            else:
                scenValue, num = scen_value(file, map_size, scen_idx, dtype)
            setime_dic[scen_name[j]] = scenValue
        map_value[map] = setime_dic
    return map_value
# Plot the results
#### Bar plots ######
def plot_results(map, scen_name, scen_dic, ptype):
    X = scen_name
    full = []
    half = []
    thir = []
    for name in (scen_name):
        full.append(scen_dic[name][0])
        half.append(scen_dic[name][1])
        thir.append(scen_dic[name][2])
    X_axis = np.arange(len(X))
    plt.figure()
    bar1 = plt.bar(X_axis - 0.2, full, 0.2, label = 'full')
    bar2 = plt.bar(X_axis + 0.0, half, 0.2, label = 'half')
    bar3 = plt.bar(X_axis + 0.2, thir, 0.2, label = 'one-third')
    plt.xticks(X_axis, X)
    for rect in bar1+bar2+bar3:
        height = rect.get_height()
        if ptype == 'Time':
            plt.text(rect.get_x() + rect.get_width() / 2.0, height,  f'{height:.2f}', ha='center', va='bottom')
            plt.xlabel('Scenarios')
            plt.ylabel('Time [seconds]')
            plt.title('Training Time Across Scenarios [%s]'%map)
        elif ptype == 'EV':
            plt.text(rect.get_x() + rect.get_width() / 2.0, height,  f'{height:.4f}', ha='center', va='bottom')
            plt.xlabel('Scenarios')
            plt.ylabel('Explain Variance')
            plt.title('Explain Variance Across Scenarios [%s]'%map)
    plt.grid(True)
    plt.legend()
    plt.show()

# Loop through each map
map_name  = ['Sepang', 'Shanghai', 'YasMarina']
map_idx   = 0
map       = map_name[map_idx]
ptype     = 'Time' # Time, EV, R2
map_size  = 3
scen_name = ['race_non-cap', 'cen_non-cap', 'race_cap',  'cen_cap']
time_idx  = [2, 90, 178]
EV_idx    = [5, 93, 181]
col_diff  = 7
row_diff  = 22
map_time  = loop_track(map_name, 'mean', time_idx, scen_name, col_diff, row_diff, kernel='RQ+LIN', kernel_id=2)
EV_value  = loop_track(map_name, 'mean', EV_idx, scen_name, col_diff, row_diff)
scenT_dic = map_time[map]
scenE_dic = EV_value[map]
for name in scen_name:
    full_data = scenE_dic[name][0]
    half_data = scenE_dic[name][1]
    thrd_data = scenE_dic[name][2]
    print('###########')
    print(name)
    print('%.2f' %(full_data/full_data*100.0) + '%')
    print('%.2f' %(half_data/full_data*100.0) + '%')
    print('%.2f' %(thrd_data/full_data*100.0) + '%')
    print('###########')
# print(scenT_dic)
plot_results(map, scen_name, scenT_dic, ptype)