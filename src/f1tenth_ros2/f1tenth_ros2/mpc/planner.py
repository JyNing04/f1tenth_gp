import os
import csv
import numpy as np
import math
import matplotlib.pyplot as plt

class MPCPlanner():
    def __init__(self, x0, path, track_name, N):
        self.map_name   = track_name[0]
        self.track_name = track_name[0] + track_name[1]
        self.path       = path
        self.currentX   = 0.0 # current position of car
        self.currentY   = 0.0
        self.currentθ   = 0.0 # heading of the car
        self.lookahead  = 0.  # Lookahead distance (m)
        self.lg         = 0.1 # Lookahead gain
        self.idx        = 0
        self.track_size = 0
        self.x0         = x0
        self.N          = N
    def construct_path(self, track_name): 
        race_track = []
        file_name  = os.path.join(self.path, track_name + '.csv')
        with open(file_name, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter =',')
            next(csv_reader)
            for waypoint in csv_reader:
                race_track.append(waypoint)
            # Force elements in racetrack to be float
            race_track = list(np.array(race_track, dtype='float'))
            # self.race_track = [[float(y) for y in x] for x in self.race_track]
            track_size = len(race_track)
            return race_track, track_size
    
    def find_nearest_goal(self, curr_x, curr_y, track_name):
        ranges = []
        race_track, _ = self.construct_path(track_name)
        for idx in range(len(race_track)):
            dist_x    = math.pow(curr_x - race_track[idx][0],2)
            dist_y    = math.pow(curr_y - race_track[idx][1],2)
            eucl_dist = math.sqrt(dist_x + dist_y)
            ranges.append(eucl_dist)
        return(ranges.index(min(ranges)))

    def projectXYOref(self):
        xref       = np.zeros([2, self.N+1]) 
        v_yaw_ref  = np.zeros([2,self.N+1]) 
        xref[:2,0] = self.x0[:2]
        v_yaw_ref[0,0] = self.x0[3] # velocity
        v_yaw_ref[1,0] = self.x0[4] # yaw
        lookahead_dist = self.lookahead + self.lg * self.x0[3]
        race_track, track_size = self.construct_path(self.track_name)
        ref_idx   = int((self.find_nearest_goal(self.x0[0], self.x0[1], self.track_name) - lookahead_dist) % track_size) # "+ lookahead_dist" for Yas Marina Curcuit
        proj_idx  = ref_idx
        for idx in range(1, self.N+1):
            idx_fwd      = (ref_idx-1) % track_size
            xref[:2,idx] = race_track[idx_fwd][:2] # start ahead of the current position
            v_yaw_ref[0,idx] = race_track[idx_fwd][2]
            x_diff       = xref[0, idx] - xref[0, idx-1]
            y_diff       = xref[1, idx] - xref[1, idx-1]
            v_yaw_ref[1,idx] = np.arctan2(y_diff, x_diff)
            ref_idx      = idx_fwd
        return v_yaw_ref, xref, proj_idx


    def projectXref(self):
        xref       = np.zeros([2, self.N+1]) 
        v_yaw_ref  = np.zeros([2,self.N+1]) 
        xref[:2,0] = self.x0[:2]
        v_yaw_ref[0,0] = self.x0[3] # velocity
        v_yaw_ref[1,0] = self.x0[4] # yaw
        race_track, track_size = self.construct_path(self.track_name)
        speed     = self.x0[3]
        lookahead_dist = self.lookahead + self.lg * speed
        ref_idx   = int((self.find_nearest_goal(self.x0[0], self.x0[1], self.track_name) - lookahead_dist) % track_size) # "+ lookahead_dist" for Yas Marina Curcuit
        proj_idx  = int((self.find_nearest_goal(self.x0[0], self.x0[1], self.track_name)) % track_size) # projection idx on centerline
        dist_ref  = np.linalg.norm(self.x0[:2] - race_track[proj_idx][:2])
        if 'raceline' in self.track_name:
            cen_track, cen_size = self.construct_path(self.map_name + '_centerline')
            cen_idx   = int((self.find_nearest_goal(self.x0[0], self.x0[1], self.map_name + '_centerline') - lookahead_dist) % cen_size)
            proj_idx  = int((self.find_nearest_goal(self.x0[0], self.x0[1], self.map_name + '_centerline')) % cen_size)
            dist_ref  = np.linalg.norm(self.x0[:2] - cen_track[proj_idx][:2])
            y_diff    = cen_track[cen_idx-1][1]-cen_track[cen_idx][1]
            x_diff    = cen_track[cen_idx-1][0]-cen_track[cen_idx][0]
            ref_yaw   = np.arctan2(y_diff, x_diff) # yaw of ref path
        else:
            idx_fwd    = (ref_idx-1) % track_size
            y_diff     = race_track[idx_fwd][1]-race_track[ref_idx][1]
            x_diff     = race_track[idx_fwd][0]-race_track[ref_idx][0]
            ref_yaw    = np.arctan2(y_diff, x_diff)
        for idx in range(1, self.N+1):
            idx_fwd      = (ref_idx-1) % track_size
            xref[:2,idx] = race_track[idx_fwd][:2] # start ahead of the current position
            v_yaw_ref[0,idx] = race_track[idx_fwd][2]
            x_diff       = xref[0, idx] - xref[0, idx-1]
            y_diff       = xref[1, idx] - xref[1, idx-1]
            v_yaw_ref[1,idx] = np.arctan2(y_diff, x_diff)
            ref_idx      = idx_fwd
        e_yaw    = self.x0[2] - ref_yaw
        pt_prj   = np.array([dist_ref*np.sin(e_yaw), dist_ref*np.cos(e_yaw)])
        dist_prj = np.linalg.norm(pt_prj)
        return v_yaw_ref, dist_prj, xref, proj_idx


# Test cases
if __name__ == '__main__':
    N          = 1100
    x0         = [-0.37384143471717834,-0.3233968913555145, 0.1, 5.5, 0.5]
    path       = os.path.expanduser('~/f1tenth_gp/src/f1tenth_ros2/f1tenth_ros2/data/tracks_velocities') 
    map_index  = 0
    map_list   = ['Sepang', 'Shanghai', 'YasMarina']
    map_name   = map_list[map_index]
    track_name = [map_name ,'_raceline_ED'] # _centerline or _raceline_ED
    planner    = MPCPlanner(x0, path, track_name, N)
    v_yaw_ref, dist_prj, xref, proj_idx = planner.projectXref()
    # v_yaw_ref, xref, proj_idx = planner.projectXYOref()
    # print(dist_prj)
    # plot reference tracjectory
    plt.figure()
    plt.plot(xref[0], xref[1], 'o-')

    plt.figure()
    plt.plot(v_yaw_ref[1,:])
    plt.show()
