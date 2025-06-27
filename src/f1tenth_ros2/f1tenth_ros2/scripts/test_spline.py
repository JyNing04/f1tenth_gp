import csv
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import CubicSpline

path       = os.path.expanduser('~/dev_ws/src/f1tenth_ros2/Tracks/Shanghai')
track_name = 'Shanghai_centerline'
file_name  = os.path.join(path, track_name + '.csv')
race_track = []
with open(file_name, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter =',')
    next(csv_reader)
    for waypoint in csv_reader:
        race_track.append(waypoint)
    # Force elements in racetrack to be float
    race_track = np.array(race_track, dtype='float')
    # self.race_track = [[float(y) for y in x] for x in self.race_track]
    # track_size = len(race_track)

x = race_track[:,0]
y = race_track[:,1]
v = race_track[:,2]
theta_0 = 0
theta = np.zeros_like(x)
for i in range(theta.shape[0]):
    if i == 0:
        theta[i] = 0
    else:
        theta[i] = theta[i-1] + v[i]*0.02
print(theta)