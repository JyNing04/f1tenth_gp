import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import math
from scipy import signal


class TrackParams():
    def __init__(self, map):
        self.map  = map
        self.path = os.path.expanduser('~/dev_ws/src/f1tenth_ros2/Tracks/' + map)
        race_file = map + "_raceline"
        cent_file = map + "_centerline"
        centerline, self.c_size = self.loading_trackfile(cent_file)
        raceline, r_size   = self.loading_trackfile(race_file)
        raceline_re        = signal.resample(raceline, self.c_size)
        self.center_x      = centerline[:, 0]
        self.center_y      = centerline[:, 1]
        self.center_line = np.concatenate([
                    self.center_x.reshape(1,-1), 
                    self.center_y.reshape(1,-1)
                    ])
        diff = np.diff(self.center_line)
        self.theta_track = np.cumsum(np.linalg.norm(diff, 2, axis=0))
        self.theta_arr   = np.concatenate([np.array([0]), self.theta_track])
    
    def loading_trackfile(self, file):
        track    = []
        file_name = os.path.join(self.path, file + '.csv')
        with open(file_name, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter =',')
                next(csv_reader)
                for waypoint in csv_reader:
                    track.append(waypoint)
                # Force elements in racetrack to be float
                race_track = np.array(track, dtype=np.float32)
                track_size = len(race_track)
                return race_track, track_size

    def Projection(self, point, line):

        assert len(point)==1
        assert len(line)==2

        x = np.array(point[0])
        x1 = np.array(line[0])
        x2 = np.array(line[len(line)-1])

        dir1 = x2 - x1
        dir1 /= np.linalg.norm(dir1, 2)
        proj = x1 + dir1*np.dot(x - x1, dir1)

        dir2 = (proj-x1)
        dir3 = (proj-x2)

        # check if this point is on the line, otw return closest vertex
        if np.linalg.norm(dir2, 2)>0 and np.linalg.norm(dir3, 2)>0:
            dir2 /= np.linalg.norm(dir2)
            dir3 /= np.linalg.norm(dir3)
            is_on_line = np.linalg.norm(dir2-dir3, 2) > 1e-10
            if not is_on_line:
                if np.linalg.norm(x1-proj, 2) < np.linalg.norm(x2-proj, 2):
                    proj = x1
                else:
                    proj = x2
        dist = np.linalg.norm(x-proj, 2)
        return proj, dist

    def project(self, x, y, raceline):
        """	finds projection for (x,y) on a raceline
        """
        point = [(x, y)]
        n_waypoints = raceline.shape[1]

        proj = np.empty([2,n_waypoints])
        dist = np.empty([n_waypoints])
        for idl in range(-1, n_waypoints-1):
            line = [raceline[:,idl], raceline[:,idl+1]]
            proj[:,idl], dist[idl] = self.Projection(point, line)
        optidx = np.argmin(dist)
        if optidx == n_waypoints-1:
            optidx = -1
        optxy = proj[:,optidx]
        return optxy, optidx


    def theta2xy(self, theta):
        idt = 0
        while idt < self.theta_arr.shape[0]-1 and self.theta_arr[idt]<=theta:
            idt += 1
        deltatheta = (theta-self.theta_arr[idt-1])/(self.theta_arr[idt]-self.theta_arr[idt-1])
        x = self.center_x[idt-1] + deltatheta*(self.center_x[idt]-self.center_x[idt-1])
        y = self.center_y[idt-1] + deltatheta*(self.center_y[idt]-self.center_y[idt-1])
        return x, y


    def xy2theta(self, x, y):
        """	finds theta on center line for a given (x,y) coordinate
        """
        track_length  = self.c_size
        optxy, optidx = self.project(x, y, self.center_line)
        distxy = np.linalg.norm(optxy-self.center_line[:,optidx],2)
        dist = np.linalg.norm(self.center_line[:,optidx+1]-self.center_line[:,optidx],2)
        deltaxy = distxy/dist
        if optidx==-1:
            theta = self.theta_track[optidx] + deltaxy*(track_length-self.theta_track[optidx])
        else:
            theta = self.theta_track[optidx] + deltaxy*(self.theta_track[optidx+1]-self.theta_track[optidx])
        theta = theta % track_length
        return theta

if __name__ == '__main__':
    map   = "Shanghai" # Shanghai, Sepang, YasMarina
    track = TrackParams(map)
    theta = track.theta_track
    print(theta)
    plt.figure()
    plt.plot(np.arange(len(theta)), theta)
    plt.show()


