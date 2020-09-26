import cv2 as cv
import numpy as np
from skimage import morphology, data, color
from sklearn.cluster import MeanShift
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from math import atan2, degrees, radians, cos, sin

class Detectors(object):
    """Detectors class to detect objects in video frame
    """
    def __init__(self,mode = 1):
        """Initialize variables used by Detectors class
        Args:
            mode:0-background subtraction using image
            mode:1-background online training
        Look:
            all points are in Fortran coordinate
        """
        #train background frames
        self.bg_train_count = 4800*2
        self.fgbg = cv.createBackgroundSubtractorMOG2(history = self.bg_train_count, varThreshold = 16, detectShadows = False)  //history：用于训练背景的帧数  varThreshold：方差阈值，用于判断当前像素是前景还是背景。一般默认16
        #train background
        self.forecount = 0
        self.firstFrame = True
        # if traing background
        self.mode = mode # 0:not train 1:train
        # crossing detection
        self.area_ratio = 1.4 # 1:1.4 s30:2, s40:1.6 new40:1.4
        # image to save 
        self.background = None
        self.thresh = 5 # binary threshold 1:5; 401:40; s40: 18; new40:15;
        self.area_per = None  # median area
        self.area_thresh = 200
        self.dilate_size = 5
        self.frame_gray = None
        self.fgmask = None
        self.fgmask_approx = None
        self.frame_and = None
        self.margin = 20
        #Meanshift
        self.bandwidth = 15 #1:15 s40:17 
        self.points_thresh = 2
        # neural network
        self.success_rate = 0.8
        self.model = load_model('./model/zebra_FLA.h5')
        self.step = 6
        self.sample_size = 40
        # orientation 
        self.approx_ratio = 0.006
        self.orient_thresh = 2000  # find orientation of single fish
        self.orient_thresh_f = 20  # filter head with two potential orientations
        self.orient_thresh_f_1 = 30 # filter head with two potential orientations that are close
        self.orient_thresh_d = 30 ** 2  # find orientation of multiple fishes
        self.orient_thresh_d_u = 130 ** 2  # upper limit
        # background address
        self.back_address = "./video/back/back1.png"


    def Get_binary(self, frame):
        """
            - Perform Background Subtraction and Get mask
            - mode:0-background subtraction using image
            - mode:1-background online training
        Args:
            frame: single video frame
        Return:
            fgmask: 
        """
                # Perform Background Subtraction
        if self.mode == 1:
            if(self.forecount< self.bg_train_count):
                fgmask = self.fgbg.apply(frame, 0.5)
                self.forecount += 1
                return None
            else:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                self.frame_gray = gray

                fgmask = self.fgbg.apply(frame, learningRate=0.0005)
                self.background = self.fgbg.getBackgroundImage()
                cv.imwrite("./video/back.jpg", self.background)
                fgmask = cv.medianBlur(fgmask,5)
                
                self.fgmask = fgmask
                return fgmask
        elif self.mode == 0:
            if self.forecount == 0:
                self.forecount += 1
                self.background = cv.imread(self.back_address)
                self.background = cv.cvtColor(self.background, cv.COLOR_BGR2GRAY)
            
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            self.frame_gray = gray

            foreground = cv.absdiff(gray, self.background)
            ret, fgmask = cv.threshold(foreground, self.thresh, 255, cv.THRESH_BINARY)
            # ret, fgmask = cv.threshold(foreground,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
            #fgmask = cv.medianBlur(fgmask,7)
            kernel = np.ones((5,1),np.uint8)
            fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
            kernel = np.ones((1,5),np.uint8)
            fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
            kernel = np.ones((self.dilate_size,self.dilate_size),np.uint8)
            fgmask = cv.morphologyEx(fgmask, cv.MORPH_DILATE, kernel)
            
            self.fgmask = fgmask

            return fgmask
    
    def Detect_single(self,contour,b_fgmask = False, b_network = True):
        """ 
            input: contour
                   b_fgmask = Flase: to use approx to calculate orientation
                   b_fgmask = Flase: to use fgmask to calculate orientation
                   b_network = True: use network to find head
            Return: point, angel(-180~180), in numpy
        """

        # Get ROI
        x_roi,y_roi,w_roi,h_roi = cv.boundingRect(contour)
        roi = np.zeros([h_roi + self.margin*2, w_roi + self.margin*2], dtype=np.uint8)
        if b_fgmask:
            roi[self.margin:h_roi + self.margin, self.margin:w_roi + self.margin] = self.fgmask[y_roi:y_roi+h_roi,x_roi:x_roi+w_roi].copy()
        else:
            mask_temp = np.zeros_like(self.fgmask)
            epsilon = self.approx_ratio * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            mask_temp = cv.drawContours(mask_temp, [approx], 0, (255), -1)
            roi[self.margin:h_roi + self.margin, self.margin:w_roi + self.margin] = mask_temp[y_roi:y_roi+h_roi,x_roi:x_roi+w_roi]
        
        roi_neural = np.zeros([h_roi + self.margin * 2, w_roi + self.margin * 2], dtype=np.uint8)
        roi_neural[self.margin:h_roi + self.margin, self.margin:w_roi + self.margin] = self.frame_and[y_roi:y_roi+h_roi,x_roi:x_roi+w_roi].copy()
        # extract centerline
        binary_bool = roi > 0
        skeleton = morphology.skeletonize(binary_bool, method='lee')
        points = np.where(skeleton>0)
        points = np.array([points[1], points[0]]).T#x,y

        # find endpoints
        kernel = np.uint8([[1,  1, 1],
                       [1,  10, 1],
                       [1,  1, 1]])
        src_depth = -1
        skeleton[np.where(skeleton>0)] = 1
        filtered = cv.filter2D(skeleton, src_depth, kernel)
        endpoints = np.where(filtered == 11)
        endpoints = np.array([endpoints[1], endpoints[0]]).T#x,y
        # find_head
        img_list = []
        if b_network:
            window_size = self.sample_size//2
            for i in range(endpoints.shape[0]):
                roi_win = roi_neural[endpoints[i][1] - window_size:endpoints[i][1] + window_size, endpoints[i][0] - window_size:endpoints[i][0] + window_size]
                # cv.imshow("2", roi_win)
                # cv.waitKey()
                test_temp = cv.resize(roi_win, (40, 40))
                test_temp = (test_temp.reshape(1, 40, 40, 1)).astype('float32') / 255
                img_list.append(test_temp)

            predict_list = tf.data.Dataset.from_tensor_slices(img_list)
            predict = self.model.predict(predict_list)
            if np.max(predict[:, 1]) < 0.2:
                return None,None,None,None

            headpoint = endpoints[np.argmax(predict[:, 1])]
            tailpoint = endpoints[np.argmin(predict[:, 1])]

        else:
            window_size = 20
            areas = []
            for i in range(endpoints.shape[0]):
                roi_win = roi[endpoints[i][1] - window_size:endpoints[i][1] + window_size, endpoints[i][0] - window_size:endpoints[i][0] + window_size]
                area = np.where(roi_win > 0)[0].shape[0]
                areas.append(area)
            areas = np.array(areas)
            headpoint = endpoints[np.argmax(areas)]

        #find orientation
        #head
        dist_thresh = self.orient_thresh
        dist = points - headpoint
        dist = dist.T[0] * dist.T[0] + dist.T[1] * dist.T[1]
        p = points[np.where(dist<dist_thresh)]
        x = np.argmax(dist[np.where(dist<dist_thresh)])
        orpoint = p[x]
        #tail
        dist_thresh = self.orient_thresh
        dist = points - tailpoint
        dist = dist.T[0] * dist.T[0] + dist.T[1] * dist.T[1]
        p = points[np.where(dist<dist_thresh)]
        x = np.argmax(dist[np.where(dist<dist_thresh)])
        orpoint_tail = p[x]        
        # coordinate system
        origin_point = np.array([x_roi-self.margin,y_roi-self.margin])
        headpoint += origin_point
        orpoint += origin_point
        
        tailpoint += origin_point
        orpoint_tail += origin_point
        # calculate angle
        angle = degrees(atan2(-(headpoint[1] - orpoint[1]), headpoint[0] - orpoint[0]))
        angle_tail = degrees(atan2(-(tailpoint[1] - orpoint_tail[1]), tailpoint[0] - orpoint_tail[0]))
        d_angle = abs(angle) + abs(angle_tail)
        bendangle = min(d_angle, 360-d_angle, key=abs)
        return headpoint,orpoint,angle,abs(bendangle)

    def Filter_skeleton(self, skeleton, mask, head_point, erode=2):
        # erode mask
        mask_temp = mask.copy()
        se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3), (-1, -1))
        mask_temp = cv.morphologyEx(mask_temp, cv.MORPH_ERODE, se, iterations=erode)
        head_point = np.array([head_point[1], head_point[0]])  #row,col coordinate
        # find points in skeleton
        point_array = np.where(skeleton>0) 
        point_array = np.array(point_array).T
        # find closest head point
        array_temp = (point_array - head_point) * (point_array - head_point)
        array_temp = array_temp.T[0] + array_temp.T[1]
        index = np.argmin(array_temp)
        head_point = point_array[index]
        #filter
        point_out = []
        for point in point_array:
            #draw line
            mask_line = np.zeros_like(mask)
            cv.line(mask_line, (point[1], point[0]), (head_point[1], head_point[0]), (255), 1)
            num_line = np.where(mask_line > 0)[0].shape[0]
            #mask_line and mask
            dst_and = cv.bitwise_and(mask, mask_line)
            num_dst = np.where(dst_and > 0)[0].shape[0]
            if (num_line == num_dst):
                point_out.append(point)
        
        mask_temp  = np.zeros_like(mask)
        point_out = np.array(point_out) # return points
        mask_temp[(point_out.T[0], point_out.T[1])] = 1  # return an image with points
        
        #find longest contour
        ids,skeleton_slice = cv.connectedComponents(mask_temp, connectivity=8, ltype=cv.CV_32S)
        head_id = skeleton_slice[head_point[0], head_point[1]]
        skeleton_slice[np.where(skeleton_slice != head_id)] = 0
        skeleton_slice[np.where(skeleton_slice == head_id)] = 1
        mask_find = skeleton_slice.astype(np.uint8)

        return mask_find
            
    def Find_and_filter_point(self, mask_find, head_points, head_point, thresh=20):
        #find endpoints
        skel = mask_find
        kernel = np.uint8([[1,  1, 1],
                        [1,  10, 1],
                        [1,  1, 1]])
        src_depth = -1
        filtered = cv.filter2D(skel, src_depth, kernel)
        endpoints = np.where(filtered == 11)
        endpoints = np.array([endpoints[1], endpoints[0]]).T  #x,y coordinate
        if endpoints.shape[0] == 0:
            return None,None
        #filter endpoints
        del_list  = []
        for point in head_points:
            array_temp = (endpoints - point) * (endpoints - point)
            array_temp = array_temp.T[0] + array_temp.T[1]
            #save_list.append(endpoints[np.where(array_temp > thresh * thresh)])
            #del_list.append(endpoints[np.where(array_temp < thresh * thresh)])
            for i in np.where(array_temp < thresh * thresh)[0]:
                del_list.append(i)
        
        array_temp = (endpoints - head_point) * (endpoints - head_point)
        array_temp = array_temp.T[0] + array_temp.T[1]
        for i in np.where(array_temp < self.orient_thresh_d)[0]:
            del_list.append(i)
        for i in np.where(array_temp > self.orient_thresh_d_u)[0]:
            del_list.append(i)

        #find unique endpoints
        del_list = np.array(del_list)
        del_list = np.unique(del_list)
        total_list = np.array(range(endpoints.shape[0]))
        out_list = np.setdiff1d(total_list, del_list)
        
        if len(out_list) > 0:
            flag = True # point is not near head
            out_points = endpoints[(out_list)]
        else:
            flag = False # point is near head
            out_points = endpoints[(del_list)]

        return out_points,flag

    def Find_direction(self,head_point, out_points, flag, head_close=False):
        if head_close==False:
            array_temp = (out_points - head_point) * (out_points - head_point)
            array_temp = array_temp.T[0] + array_temp.T[1]

            if array_temp.shape[0] > 1:
                sort_temp = np.argsort(array_temp)
                d1 = np.sqrt(array_temp[sort_temp[-1]])
                d2 = np.sqrt(array_temp[sort_temp[-2]])
                d3 = (out_points[sort_temp[-1]] - out_points[sort_temp[-2]]) * (out_points[sort_temp[-1]] - out_points[sort_temp[-2]])
                d3 = np.sqrt(d3.T[0] + d3.T[1])
                if abs(d1-d2) < self.orient_thresh_f and d3 > self.orient_thresh_f_1:
                    out_points = out_points[[sort_temp[-1], sort_temp[-2]]]
                    return out_points
                else:
                    out_point = out_points[sort_temp[-1]]
                    return out_point        
            else:
                index = np.argmax(array_temp)
                out_point = out_points[index]
                return out_point
        else:
            array_temp = (out_points - head_point) * (out_points - head_point)
            array_temp = array_temp.T[0] + array_temp.T[1]
            #indexes = array_temp.argsort()[-1:(-1 - head_number):-1]
            indexes = np.where(array_temp > self.orient_thresh_d)
            if indexes[0].shape[0]>0:
                out_points = out_points[indexes]
            else:
                #indexes = array_temp.argsort()[-1:(-1 - 2):-1]
                index = array_temp.argsort()[-1]
                out_points = out_points[index]
            return out_points

    def Find_crossing_direction(self, roi_mask, headpoints, num=2):
        # find skeleton
        binary_bool = roi_mask > 0
        skeleton = morphology.skeletonize(binary_bool, method='lee')
        
        orient_points = []
        head_outpoints = []
        for headpoint in headpoints:
            # filter skeleton
            mask_find = self.Filter_skeleton(skeleton, roi_mask, headpoint, erode=3)
            out_points, flag = self.Find_and_filter_point(mask_find, headpoints, headpoint)
            if out_points is None:
                continue
            if num == 1:
                orient_points = self.Find_direction(headpoint, out_points, flag, head_close=True)
                head_outpoints = headpoints*len(orient_points)
                break
            else:
                out_point = self.Find_direction(headpoint, out_points, flag)
                if out_point.ndim > 1:
                    orient_points.append(out_point[0])
                    orient_points.append(out_point[1])
                    head_outpoints.append(headpoint)
                    head_outpoints.append(headpoint)
                else:
                    orient_points.append(out_point)
                    head_outpoints.append(headpoint)        
        return head_outpoints, orient_points

    
    def Detect_crossing(self, contour, b_fgmask=False):
        """ 
            input: contour
                   b_fgmask = Flase: to use approx to calculate orientation
                   b_fgmask = Flase: to use fgmask to calculate orientation
            Return: points, angels(-180~180), in numpy
        """
        
        # get roi and roi mask
        x_roi,y_roi,w_roi,h_roi = cv.boundingRect(contour)
        roi = np.zeros([h_roi + self.margin * 2, w_roi + self.margin * 2], dtype=np.uint8)
        roi[self.margin:h_roi + self.margin, self.margin:w_roi + self.margin] = self.frame_and[y_roi:y_roi+h_roi,x_roi:x_roi+w_roi].copy()
        roi_mask = np.zeros([h_roi + self.margin * 2, w_roi + self.margin * 2], dtype=np.uint8)
        roi_mask[self.margin:h_roi + self.margin, self.margin:w_roi + self.margin] = self.fgmask_approx[y_roi:y_roi+h_roi,x_roi:x_roi+w_roi].copy()
        #cv.imwrite("./crossing/"+str(count)+'.jpg', roi)
        # sample roi
        img_list = []
        point_list = []
        for x in range(0, w_roi + self.margin * 2-self.sample_size, self.step):
            for y in range(0, h_roi + self.margin * 2 - self.sample_size, self.step):
                if roi[y+self.sample_size//2 , x+self.sample_size//2] > 0: 
                    test_temp = roi[y:y + self.sample_size, x:x + self.sample_size].copy()
                    if self.sample_size != 40:
                        test_temp = cv.resize(test_temp, (40, 40))
                    test_temp = (test_temp.reshape(1, 40, 40, 1)).astype('float32') / 255
                    img_list.append(test_temp)
                    point_list.append([x+self.sample_size//2,y+self.sample_size//2])#x,y
                else:
                    pass
        # find heads
        predict_list = tf.data.Dataset.from_tensor_slices(img_list)
        predict = self.model.predict(predict_list)
        result = predict[:,1] > self.success_rate
        point_list = np.array(point_list)
        p_list = point_list[result]

        if p_list.shape[0] == 0:
            return (x_roi,y_roi,w_roi,h_roi),None,None,None
        num = 2
        # Meanshift
        clf = MeanShift(bandwidth=self.bandwidth)
        predicted=clf.fit_predict(p_list)
        headpoints = []
        if np.max(predicted) == 0:
            mean = np.mean(p_list, axis=0)
            headpoints.append(mean)
            num = 1
        else:
            for index in range(0,np.max(predicted)+1):
                if np.count_nonzero(predicted==index) > self.points_thresh:
                    points_cla = p_list[predicted==index]
                    mean = np.mean(points_cla,axis=0)
                    headpoints.append(mean)
        
        # find orientation
        if num ==1:
            head_outpoints, orient_points = self.Find_crossing_direction(roi_mask, headpoints, num=1)
            headpoints = np.array(head_outpoints).reshape(-1,2)
        else:
            head_outpoints, orient_points = self.Find_crossing_direction(roi_mask, headpoints)
            headpoints = np.array(head_outpoints).reshape(-1,2)
        
        #####Debug######
        # for i in range(len(orient_points)):
        #     cv.circle(roi, (int(headpoints[i][0]), int(headpoints[i][1])), 6, 255, -1)
        #     cv.circle(roi, (orient_points[i][0], orient_points[i][1]), 10, 255, 2)
        #     cv.line(roi, (int(headpoints[i][0]), int(headpoints[i][1])), (orient_points[i][0], orient_points[i][1]), (0, 255, 0), 3)  #线
        # roi = cv.pyrUp(roi) 
        # cv.imshow("roi", roi)
        # cv.waitKey(100)

        # coordinate system
        origin_point = np.array([x_roi-self.margin,y_roi-self.margin])
        orient_points = np.array(orient_points).reshape(-1,2)
        headpoints += origin_point
        orient_points += origin_point
        # Filter short lines
        array_temp = (headpoints - orient_points) * (headpoints - orient_points)
        array_temp = array_temp.T[0] + array_temp.T[1]
        indexes = np.where(array_temp > self.orient_thresh_d)
        if indexes[0].shape[0] > 0:
            headpoints = headpoints[indexes]
            orient_points = orient_points[indexes]

        ##### calculate angle #####
        sub = headpoints - orient_points
        angles = []
        for i in range(sub.shape[0]):
            angle = degrees(atan2(-sub[i][1], sub[i][0]))
            angles.append(angle)
        angles = np.array(angles)
        return (x_roi,y_roi,w_roi,h_roi),headpoints,orient_points,angles
       
    def Detect(self, frame):
        """
            - get position and orientation
        Args:
            frame: single video frame
        Return:
            position: 
            orientation:
        """
        fgmask = self.Get_binary(frame)
        contours, hierarchy = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        ###filter contours###
        if self.firstFrame:
            self.firstFrame = False
            areas = []
            for i in range(len(contours)):
                if cv.contourArea(contours[i])>self.area_thresh:
                    areas.append([cv.contourArea(contours[i])])
            areas = np.array(areas)
            self.area_per = np.median(areas)
            print(self.area_per)
        
        ###group contour###
        single = []
        crossing = []

        mask = np.zeros_like(fgmask)
        ratio = self.approx_ratio
        for i in range(len(contours)):
            area = cv.contourArea(contours[i])
            area_ratio = self.area_ratio
            if (self.area_per * 0.5) < area < (self.area_per * area_ratio):
                single.append(i)
                
                epsilon = ratio * cv.arcLength(contours[i], True)
                approx = cv.approxPolyDP(contours[i], epsilon, True)
                cv.drawContours(mask, [approx], 0, (255), -1)

            elif (self.area_per * area_ratio) < area:
                crossing.append(i)

                epsilon = ratio * cv.arcLength(contours[i], True)
                approx = cv.approxPolyDP(contours[i], epsilon, True)
                cv.drawContours(mask, [approx], 0, (255), -1)

        ###update mask###
        self.fgmask_approx = mask
        self.frame_and = cv.bitwise_and(self.frame_gray, self.fgmask)
        ###detect###
        positions = []
        orientations = []
        cross_states = []
        bend_angles = []
        for i in single:
            headpoint, orpoint, angle, bend_angle = self.Detect_single(contours[i])
            # cv.circle(frame, (headpoint[0], headpoint[1]), 6, (0, 0, 255), -1)
            # cv.circle(frame, (orpoint[0], orpoint[1]), 4, (255, 0, 0), -1)
            # cv.line(frame, (headpoint[0], headpoint[1]), (orpoint[0], orpoint[1]), (0, 255, 0), 3)  #线
            if headpoint is None:
                continue
            positions.append(headpoint)
            orientations.append(angle)
            cross_states.append(0)
            bend_angles.append(bend_angle)

        if len(crossing) > 0:
            for i in crossing:
                (x, y, w, h), headpoints, orient_points, angles = self.Detect_crossing(contours[i])
                if headpoints is None:
                    continue
                frame = cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                positions += list(headpoints)
                orientations += list(angles)
                cross_states += [1] * len(list(headpoints))
                bend_angles  += [-1] * len(list(headpoints))
        #output
        points = []
        for i in range(len(positions)): 
            x = positions[i][0]
            y = positions[i][1]
            pos = np.array([[x], [y]])
            points.append(np.round(pos))
        outputs = []
        for i in range(len(points)):
            outputs.append([points[i],orientations[i],cross_states[i],bend_angles[i]])

        return outputs, frame, positions, orientations, bend_angles
    
    
    def Detect_simple(self, frame):
        fgmask = self.Get_binary(frame)
        contours, hierarchy = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        ###filter contours###
        if self.firstFrame:
            self.firstFrame = False
            areas = []
            for i in range(len(contours)):
                if cv.contourArea(contours[i])>self.area_thresh:
                    areas.append([cv.contourArea(contours[i])])
            areas = np.array(areas)
            self.area_per = np.median(areas)
            print(self.area_per)
        
        self.frame_and = cv.bitwise_and(self.frame_gray, self.fgmask)
        #detect 
        positions = []
        orientations = []
        cross_states = []
        bend_angles = []
        outputs = []

        for i in range(len(contours)):
            area = cv.contourArea(contours[i])
            if (self.area_per * 0.5) < area: 
                cnt = contours[i]
                M = cv.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                point = np.array([cx, cy])
                pos = np.array([[cx], [cy]])
                pos = np.round(pos)

                positions.append(point)
                orientations.append(0)
                cross_states.append(0)
                bend_angles.append(0)

                outputs.append([pos, 0, 0, 0])
        
        return outputs, frame, positions, orientations, bend_angles   
            



def Draw(frame,positions, orientaions):
    for i in range(len(positions)):
        degree = orientaions[i]
        position = positions[i]
        position = position.astype(np.int)
        dist = 30
        dist1 = 6
        anchor = np.array([position[0]-dist*cos(radians(degree)),position[1]+dist*sin(radians(degree))])
        p1 = np.array([anchor[0]-dist1*sin(radians(degree)),anchor[1]-dist1*cos(radians(degree))])
        p2 = np.array([anchor[0]+dist1*sin(radians(degree)),anchor[1]+dist1*cos(radians(degree))])
        p1 = p1.astype(np.int)
        p2 = p2.astype(np.int)

        frame = cv.polylines(frame,[np.array([p1,p2,position])],True,(255,0,0),3)
        frame = cv.circle(frame, (position[0], position[1]), 10, (0, 0, 255), -1)
    return frame

def Draw_single(frame,position, orientation,color,dist = 40, dist1 = 5, b_point = True, b_draw_ang = True):
    degree = orientation
    position = position.reshape(-1)
    position = position.astype(np.int)
    dist = dist
    dist1 = dist1
    anchor = np.array([position[0]-dist*cos(radians(degree)),position[1]+dist*sin(radians(degree))])
    p1 = np.array([anchor[0]-dist1*sin(radians(degree)),anchor[1]-dist1*cos(radians(degree))])
    p2 = np.array([anchor[0]+dist1*sin(radians(degree)),anchor[1]+dist1*cos(radians(degree))])
    p1 = p1.astype(np.int)
    p2 = p2.astype(np.int)
    if b_point:
        frame = cv.circle(frame, (position[0], position[1]), 10, (34, 221, 201), 2)
        frame = cv.circle(frame, (position[0], position[1]), 15, (18, 188, 0), 2)
    if b_draw_ang:
        frame = cv.polylines(frame, [np.array([p1, p2, position])], True, color, 4)
    return frame
if __name__ == "__main__":
    cap = cv.VideoCapture('./video/data/new40.avi')
    detector = Detectors(mode=0)
    count_start = 79
    count = 0
    # Infinite loop to process video frames
    while (True):
        count = count + 1
        ret, frame = cap.read()
        if frame is None:
            break
            # Make copy of original frame
            # Detect and return centeroids of the objects in the frame
        if detector.mode == 1:
            detector.Get_binary(frame)
        if count > count_start and (detector.mode == 0):
            print(count)
            e1 = cv.getTickCount()

            _, frame, positions, orientaions, bend_angles = detector.Detect(frame)
            
            e2 = cv.getTickCount()
            time = (e2 - e1) / cv.getTickFrequency()
            print(time)
            origin = frame.copy()
            frame = Draw(frame, positions, orientaions)
            cv.namedWindow("fgmask", cv.WINDOW_NORMAL)
            cv.imshow("fgmask", detector.fgmask)

            # cv.namedWindow("origin", cv.WINDOW_NORMAL)
            # cv.imshow("origin", origin)
            cv.namedWindow("frame", cv.WINDOW_NORMAL)
            cv.imshow("frame", frame)
            # cv.imwrite("./output/"+'1'+"/and/" + str(count) + ".jpg", detector.frame_and)
            cv.waitKey()
            # cv.imwrite("./output/"+'1'+"/result/"+str(count)+'.jpg',frame) 
        if count == detector.bg_train_count:
            print("True")
             

    print(count)
    cv.destroyAllWindows()