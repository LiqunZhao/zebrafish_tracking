# Import python libraries
import cv2 as cv
import numpy as np
import pandas as pd
import copy
from detectors import Detectors, Draw_single
from tracker_adapt import Tracker



def main():
    """Main function for multi object tracking
    Usage:
        $ python3.6.8 objectTracking.py
    Pre-requisite:
        - Python3.6.8
        - Numpy
        - SciPy
        - Opencv 4.1.0.25 for Python
    Args:
        None
    Return:
        None
    """

    # Create opencv video capture object
    directory = '1'
    count_start = 40
    ratio = 2 / 98
    count_end = count_start + 800

    cap = cv.VideoCapture("./video/data/"+directory+".avi")
    output_data = pd.DataFrame()                                          //读写csv
    # Create Object Detector
    detector = Detectors(mode=0)                                          //0-background subtraction using image  1-background online training

    # Create Object Tracker
    ang_mode = 1
    tracker = Tracker(dist_thresh = 200, ang_thresh=120, max_frames_to_skip=5, max_trace_length=3, trackIdCount=0,ang_mode=ang_mode)



    # Variables initialization
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                    (255, 255, 0), (0, 255, 255), (255, 0, 255),
                    (235, 14, 192), (0, 69, 255), (0, 252, 100),
                    (58, 53, 159), (240, 94, 28), (145, 173, 112),
                    (0, 0, 0), (62, 129, 27), (129, 172, 93),
                    (81, 125, 34), (104, 163, 233), (45, 128, 199),
                    (0, 225, 34), (104, 0, 164), (45, 60, 0), (45, 0, 199),
                    (176, 224, 230), (65, 105, 225), (106, 90, 205), (135, 206, 235),
                    (56, 94, 15), (8, 46, 84), (127, 225, 222), (64, 24, 208),
                    (127, 225, 0), (61, 145, 64), (0, 201, 87), (34, 139, 34),
                    (124, 252, 0), (50, 205, 50), (189, 252, 201), (107, 142, 35),
                    (48, 128, 20), (255, 153, 18), (235, 142, 85), (255, 227, 132),
                    (218, 165, 205), (227, 168, 205), (255, 97, 0), (237, 145, 33)]*2
    //pause = False
    count = 0
    pd_index = 0

    # Infinite loop to process video frames
    while (True):
        count = count + 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        # When everything done, release the capture
        if (frame is None):
            cap.release()
            cv.destroyAllWindows()
        # Make copy of original frame
        orig_frame = copy.copy(frame)

        # If centroids are detected then track them
        if ( count_start < count < count_end):
            # Detect and return centeroids of the objects in the frame
            detections, _, _, _, _ = detector.Detect(frame)
            # Track object using Kalman Filter
            tracker.Update(detections)
            # For identified object tracks draw tracking line
            # Use various colors to indicate different track_id
            font = cv.FONT_HERSHEY_SIMPLEX
            
            blank = np.zeros_like(frame)
            blank += 255
            x_margin = blank.shape[1]
            y_margin = blank.shape[0]
            for i in range(len(tracker.tracks)):
                if (len(tracker.tracks[i].trace) > 1):
                    for j in range(len(tracker.tracks[i].trace)-1):
                        # Draw trace line
                        x1 = tracker.tracks[i].trace[j][0][0]
                        y1 = tracker.tracks[i].trace[j][1][0]
                        x2 = tracker.tracks[i].trace[j+1][0][0]
                        y2 = tracker.tracks[i].trace[j+1][1][0]
                        #clr = tracker.tracks[i].track_id % len(track_colors)
                        clr = i
                        cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                 track_colors[clr], 2)
                x3 = tracker.tracks[i].trace[-1][0][0]
                y3 = tracker.tracks[i].trace[-1][1][0]
                x3_r = x3
                y3_r = y3
                if x3 < 50:
                    x3 = 60
                if y3 < 50:
                    y3 = 50
                if x3 > x_margin-250:
                    x3 =x_margin-250
                
                x3 = int(x3)
                y3 = int(y3)
                len_x = len(str(int(x3_r * ratio)))
                len_y = len(str(int(y3_r * ratio)))
                len_deg = len(str(int(tracker.tracks[i].angles[-1])))
                str_x = str(x3_r * ratio)[:len_x + 2]
                str_y = str(y3_r * ratio)[:len_y + 2]
                str_deg = str(tracker.tracks[i].angles[-1])[:len_deg + 2] 

                #clr = tracker.tracks[i].track_id % len(track_colors)
                clr = tracker.tracks[i].track_id
                if tracker.tracks[i].b_detected[-1]:
                    frame = Draw_single(frame, tracker.tracks[i].trace_raw[-1], tracker.tracks[i].angles[-1], track_colors[clr], b_point=tracker.tracks[i].b_detected[-1],b_draw_ang = ang_mode)
                else:
                    frame = Draw_single(frame, tracker.tracks[i].position, tracker.tracks[i].angles[-1], track_colors[clr], b_point=tracker.tracks[i].b_detected[-1],b_draw_ang = ang_mode)
                
                frame = cv.putText(frame, "Fish" + str(clr) + ":", (int(x3) - 60, int(y3) - 20), font, 0.7, (215, 0, 0), 2)
                if ang_mode == 1:                                           
                    frame = cv.putText(frame, "(" + str_x + "mm," + str_y + "mm," + str_deg + "deg)", (int(x3) + 20, int(y3) - 20), font, 0.6, (0, 165, 255), 2)
                else:
                    frame = cv.putText(frame, "(" + str_x + "mm," + str_y + "mm)", (int(x3) + 20, int(y3) - 20), font, 0.6, (0, 165, 255), 2)
                
                blank = Draw_single(blank, tracker.tracks[i].trace[-1], tracker.tracks[i].angles[-1], track_colors[clr], dist=60, dist1=15, b_point=False)
                blank = cv.putText(blank, "Fish"+str(i)+":", (int(x3)-60, int(y3)-20), font, 0.7, (215, 0, 0), 2)                                           
                blank = cv.putText(blank, "(" + str_x + "mm," + str_y + "mm," + str_deg + "deg)", (int(x3) + 20, int(y3) - 20), font, 0.6, (0, 165, 255), 2)
                
                # frame = cv.putText(frame, "Fish"+str(i), (int(x3)-30, int(y3)-20), font, 0.7, (215, 0, 0), 2) 
            #write image
            # cv.imwrite("./output/new40/mask/" + str(count) + ".jpg", detector.fgmask)
            cv.imwrite("./output/"+directory+"/and/" + str(count) + ".jpg", detector.frame_and)
            cv.imwrite("./output/"+directory+"/illu/" + str(count) + ".jpg", blank)
            cv.imwrite("./output/"+directory+"/final/" + str(count) + ".jpg", frame)
            # cv.imwrite("./output/no text/" + str(count) + ".jpg", frame)

            frame = cv.putText(frame, str(count), (50, 50), font, 1, (0, 0, 255), 2)
            # Display the resulting tracking frame
            cv.namedWindow('Tracking',cv.WINDOW_NORMAL)
            cv.imshow('Tracking', frame)
            # cv.imshow('blank', blank)
            # cv.imshow('mask', detector.fgmask)
            # cv.imshow('and', detector.frame_and)
            cv.waitKey(1)
            #cv.imwrite("./output/" + "frame" + str(count) + ".jpg", frame)
            
            # write data 
            for i in range(len(tracker.tracks)):
                trace = np.array(tracker.tracks[i].trace_save).reshape(-1, 2)
                trace_x = trace[-1, 0]
                trace_y = trace[-1, 1]
                trace_raw = np.array(tracker.tracks[i].trace_raw).reshape(-1, 2)
                trace_raw_x = trace_raw[-1, 0]
                trace_raw_y = trace_raw[-1, 1]
                angle = np.array(tracker.tracks[i].angles)[-1]
                bendangle = np.array(tracker.tracks[i].bendangles)[-1]
                state = np.array(tracker.tracks[i].states)[-1]
                b_detected = np.array(tracker.tracks[i].b_detected)[-1]
                pd_frame = len(tracker.tracks[i].angles)
                t_id = tracker.tracks[i].track_id
                fish_dict = {"Position_x": trace_x,
                "Position_y": trace_y,
                "Detection_x": trace_raw_x,
                "Detection_y": trace_raw_y,
                "Orientation": angle,
                "Bendangles":bendangle,
                "State": state,
                "b_detected": b_detected,
                "Frame": pd_frame,
                "ID": t_id}
                fish_data = pd.DataFrame(fish_dict,index=[pd_index])
                pd_index += 1 
                output_data = pd.concat([output_data, fish_data])
            output_data.to_csv("./output/"+directory+"/data.csv")

        elif count > count_end:
            break
        else:
            pass
            #cv.imwrite("./output/" + "frame" + str(count) + ".jpg", frame)




if __name__ == "__main__":
    # execute main
    main()



        # output_data = pd.DataFrame()
        # for i in range(len(tracker.tracks)):
        #     trace = np.array(tracker.tracks[i].trace_save).reshape(-1, 2)
        #     trace_x = trace[:, 0]
        #     trace_y = trace[:, 1]
        #     trace_raw = np.array(tracker.tracks[i].trace_raw).reshape(-1, 2)
        #     trace_raw_x = trace_raw[:, 0]
        #     trace_raw_y = trace_raw[:, 1]
        #     angles = np.array(tracker.tracks[i].angles)[:-1]
        #     bendangles = np.array(tracker.tracks[i].bendangles)[:-1]
        #     states = np.array(tracker.tracks[i].states)[:-1]
        #     b_detected = np.array(tracker.tracks[i].b_detected)[:-1]
        #     frames = np.array(range(len(tracker.tracks[i].angles)))[:-1]
        #     ids = np.zeros((len(tracker.tracks[i].angles)), np.uint)[:-1]
        #     ids.fill(i)
        #     fish_dict = {"Position_x": trace_x,
        #                 "Position_y": trace_y,
        #                 "Detection_x": trace_raw_x,
        #                 "Detection_y": trace_raw_y,
        #                 "Orientation": angles,
        #                 "Bendangles":bendangles,
        #                 "State": states,
        #                 "b_detected": b_detected,
        #                 "Frame": frames,
        #                 "ID":ids}
        #     fish_data = pd.DataFrame(fish_dict)
        #     output_data = pd.concat([output_data, fish_data])
        # output_data.to_csv('data.csv')   