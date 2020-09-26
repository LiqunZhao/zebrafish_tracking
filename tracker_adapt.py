# Import python libraries
import numpy as np
from adaptive_filter import AdaptiveFilter
from common import dprint
from scipy.optimize import linear_sum_assignment



class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, detection, trackIdCount):
        """Initialize variables used by Track class
        Args:
            position: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        self.AKF = AdaptiveFilter(detection[0])  # KF instance to track this object
        self.position = np.asarray(detection[0])  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = [self.position]  # trace path
        self.trace_save = [self.position]  # trace path
        self.trace_raw = [self.position]  # trace path
        self.angles = [detection[1]]  #orientation
        self.bendangles = [detection[3]]
        self.states = [detection[2]] #if cross
        self.b_detected = [True]#if detected


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self,  dist_thresh, ang_thresh, max_frames_to_skip, max_trace_length,trackIdCount,ang_mode = 1):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.ang_thresh = ang_thresh # max turn angle
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount
        self.ang_mode = ang_mode # O: dist 1: dist+angle


    def Update(self, detections):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """

        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost_dist = np.zeros(shape=(N, M))
        cost_dir = np.zeros(shape=(N, M))
        cost = np.zeros(shape=(N, M))
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                try:
                    diff = self.tracks[i].position - detections[j][0]
                    distance = np.sqrt(diff[0][0] * diff[0][0] +
                                       diff[1][0] * diff[1][0])
                    d_angle = self.tracks[i].angles[-1] - detections[j][1]
                    diff_angle = min(d_angle, d_angle+360, d_angle-360, key=abs)                    
                    # if self.tracks[i].states[-1] == 1 and detections[j][2]==1:
                    #     if abs(diff_angle) > self.ang_thresh:
                    #         diff_angle = self.dist_thresh*2
                    cost_dist[i][j] = distance
                    cost_dir[i][j] = abs(diff_angle)
                    if self.ang_mode == 1:
                        cost[i][j] = distance + abs(diff_angle)
                    else:
                        cost[i][j] = distance
                except:
                    pass

        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]
        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if self.ang_mode == 1:
                    cond = (cost_dist[i][assignment[i]] > self.dist_thresh) or (cost_dir[i][assignment[i]] > self.ang_thresh)
                else:
                    cond = cost_dist[i][assignment[i]] > self.dist_thresh
                if (cond):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        tracks_new = []
        assignment_new =[]
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames < self.max_frames_to_skip):
                tracks_new.append(self.tracks[i])
                assignment_new.append(assignment[i])
        self.tracks = tracks_new
        assignment = assignment_new

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
                if i not in assignment:
                    un_assigned_detects.append(i)

        # Start new tracks
        if(len(un_assigned_detects) != 0):
            for ii in range(len(un_assigned_detects)):
                if detections[un_assigned_detects[ii]][2]!=1:
                    track = Track(detections[un_assigned_detects[ii]], self.trackIdCount) 
                    self.trackIdCount += 1
                    self.tracks.append(track)
                else:
                    continue

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].AKF.predict()
            if (assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].AKF.update(detections[assignment[i]][0])
                x = self.tracks[i].AKF.pos[0]
                y = self.tracks[i].AKF.pos[1]
                self.tracks[i].position = np.array([[x], [y]])
                self.tracks[i].trace_save.append(self.tracks[i].position)
                self.tracks[i].trace_raw.append(detections[assignment[i]][0])
                self.tracks[i].b_detected.append(True)
                # self.tracks[i].trace_save.append(detections[assignment[i]][0])
                self.tracks[i].angles.append(detections[assignment[i]][1])
                self.tracks[i].states.append(detections[assignment[i]][2])
                self.tracks[i].bendangles.append(detections[assignment[i]][3])
            else:
                # x = self.tracks[i].AKF.pos[0]
                # y = self.tracks[i].AKF.pos[1]
                # self.tracks[i].position = np.array([[x], [y]])
                self.tracks[i].position = self.tracks[i].trace_raw[-1]
                self.tracks[i].trace_save.append(self.tracks[i].position)
                self.tracks[i].trace_raw.append(self.tracks[i].trace_raw[-1])
                self.tracks[i].b_detected.append(False)
                self.tracks[i].angles.append(self.tracks[i].angles[-1])
                self.tracks[i].states.append(-1)
                self.tracks[i].bendangles.append(-1)



            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) -
                               self.max_trace_length):
                    del self.tracks[i].trace[j]
            # if (len(self.tracks[i].position[0]) > 1):
            #     temp = np.zeros((2, 1))
            #     temp[0] = self.tracks[i].position[0][0]
            #     temp[1] = self.tracks[i].position[0][1]
            #     self.tracks[i].trace.append(temp)
            #     self.tracks[i].KF.lastResult = temp
            #     self.tracks[i].trace_save.append(temp)
            # else:
            #self.tracks[i].trace.append(self.tracks[i].position)
            self.tracks[i].trace.append(self.tracks[i].trace_raw[-1])
            #self.tracks[i].trace_save.append(self.tracks[i].position)
            # if (len(self.tracks[i].trace_save) > 5):
            #     save = np.array(self.tracks[i].trace_save).reshape(-1,2)
            #     np.savetxt('./track_result/'+str(self.tracks[i].track_id)+'.txt',save)


