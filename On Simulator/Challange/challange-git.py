
from numpy.core.function_base import linspace
import time
import cv2
import numpy as np
import AVISEngine

# from SlidingWindow import colorThresh

def warp(img):
    pts1 = np.float32([[57,145], [210,145], [0,195], [255,195]])
    pts2 = np.float32([[0,0], [250,0], [0,350], [250,350]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped_img = cv2.warpPerspective(np.float32(img), matrix, (250,350))
    # cv2.imshow('warped img', warped_img)
    # cv2.waitKey(0)
    # plt.imshow(warped_img)
    # plt.show()
    return warped_img


def colorThreshold(warped_img):
    channel = warped_img[:, :, 0]
    threshold = (170, 255)
    output = np.zeros_like(channel)
    output[(channel>=threshold[0]) & (channel<=threshold[1])]=255
    return output


def yellowDetection(warped_img):
    lower = np.uint8([190, 190,   0])
    upper = np.uint8([255, 255, 130])
    yellow_mask = cv2.inRange(warped_img, lower, upper)
    return yellow_mask


def canny(warped_img):
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray, (5,5), 0)
    canny_img = cv2.Canny(blurred_img, 50, 150) 
    return canny_img   


def combine(colorThreshold_img, canny_img):
    output = np.zeros_like(canny_img)
    output[(canny_img==255) | (colorThreshold_img==255)] = 255
    return output


def histogram(binary_img):
    image = binary_img/255
    buttom_half = image[ image.shape[0]//2:, : ]
    hist = np.sum( buttom_half, axis=0)
    return hist


def oneOrTwo(binary_img):
    hist = histogram(binary_img)
    midpoint = np.int32(hist.shape[0]//2)
    leftx_base = np.argmax(hist[:midpoint])
    rightx_base = np.argmax(hist[midpoint:]) + midpoint
    print('left=',hist[leftx_base])
    print('right=',hist[rightx_base])
    line = 1
    if ((hist[rightx_base]<80) | (hist[leftx_base]<80)):
        line = 1
    else:
        line = 2
    print('line=',line)
    return line


def yellowLineLoc(yellow_binary_img):
    # global yellowLine
    hist = histogram(yellow_binary_img)
    base = np.argmax(hist)
    midpoint = binary_img.shape[1]//2
    # print('yellow nonzero=', hist[base])
    if (hist[base]>100):
        if (base<=midpoint):
            yellowLine = np.int32(0)
        elif (base>midpoint):
            yellowLine = np.int32(1)
    else:
        yellowLine = np.int32(2)
    return yellowLine

def outOfRoadLane(binary_img, yellow_binary_img):
    yellow_line_num = yellowLineLoc(yellow_binary_img)
    if (yellow_line_num == 2):
        hist = histogram(binary_img)
        base = np.argmax(hist)
        midpoint = binary_img.shape[1]//2
        if (base<=midpoint):
            whiteLine = np.int32(0)
        elif (base>=midpoint):
            whiteLine = np.int32(1)
    else:
        whiteLine = np.int32(2)
    return yellow_line_num, whiteLine



def findLanePixels2(binary_img):
    img_copy = binary_img.copy()
    hist = histogram(binary_img)
    midpoint = np.int32(hist.shape[0]//2)
    left_base = np.argmax(hist[:midpoint])
    right_base = np.argmax(hist[midpoint:]) + midpoint

    # print('left b=',hist[left_base])
    # print('right b=',hist[right_base])

    leftx_current = left_base
    rightx_current = right_base

    min_pixels = 10
    margin = 25
    nwindows = 8
    window_height =np.int32(binary_img.shape[0]//nwindows)

    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = []
    right_lane_inds = []
    left_rec, right_rec = 1, 1
    for window in range (nwindows):
        win_y_low = binary_img.shape[0] - ((window+1)*window_height)
        win_y_high = binary_img.shape[0] - ( window*window_height)
        if (leftx_current-margin<0):
            win_xleft_low=0
        else:    
            win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        if (rightx_current+margin>binary_img.shape[1]):
            win_xright_high = binary_img.shape[1]
        else:     
            win_xright_high = rightx_current + margin

        if (left_rec != 0):
            cv2.rectangle( binary_img, (win_xleft_low , win_y_low), (win_xleft_high , win_y_high), (255,0,0), 2)
        if (right_rec != 0):
            cv2.rectangle( binary_img, (win_xright_low , win_y_low), (win_xright_high , win_y_high), (255,0,0), 2)


        good_left_inds = ((nonzeroy>=win_y_low) & (nonzeroy<win_y_high) & (nonzerox>=win_xleft_low) & (nonzerox<win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy>=win_y_low) & (nonzeroy<win_y_high) & (nonzerox>=win_xright_low) & (nonzerox<win_xright_high)).nonzero()[0]

        if ((len(good_left_inds) > min_pixels) & (len(good_right_inds) > min_pixels)):
            leftx_current = np.int32( np.mean( nonzerox[good_left_inds]))
            left_lane_inds.append(good_left_inds)
            rightx_current = np.int32( np.mean( nonzerox[good_right_inds]))
            right_lane_inds.append(good_right_inds)
            # print('1')
        elif ((len(good_left_inds) > min_pixels) & (len(good_right_inds) < min_pixels)):  
            leftx_current = np.int32( np.mean( nonzerox[good_left_inds]))
            left_lane_inds.append(good_left_inds)
            right_rec = 0
            # print('2')
        elif ((len(good_left_inds) < min_pixels) & (len(good_right_inds) > min_pixels)):   
            rightx_current = np.int32( np.mean( nonzerox[good_right_inds]))
            right_lane_inds.append(good_right_inds) 
            left_rec = 0
            # print('3')    
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass    
   
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    out_img = np.dstack((binary_img, binary_img, binary_img))
    
    return leftx, lefty, rightx, righty, out_img



def findLanePixels1(binary_img):
    hist = histogram(binary_img)
    base = np.argmax(hist)
    midpoint = binary_img.shape[1]//2
    if (base<midpoint):
        movepix = np.int32(120)
    else:
        movepix = np.int32(-120)
    nwindows = 8
    margin = 25
    minpix = 30

    window_height = np.int32(binary_img.shape[0]//nwindows)
    x_current = base

    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    lane_inds = []
    rec = 1
    for window in range (nwindows):
        win_y_low = binary_img.shape[0] - ((window+1)*window_height)
        win_y_high = binary_img.shape[0] - ( window*window_height)
        win_x_low = x_current - margin
        win_x_high = x_current + margin
        
        if (rec != 0):
            cv2.rectangle( binary_img, (win_x_low , win_y_low), (win_x_high , win_y_high), (255,0,0), 2)

        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        
        if ((len(good_inds) > minpix)):
            x_current = np.int32( np.mean( nonzerox[good_inds]))
            lane_inds.append(good_inds)
        else:
            rec = 0

    try:
        lane_inds = np.concatenate(lane_inds)
    except ValueError:
        pass
    
    linex = nonzerox[lane_inds]
    liney = nonzeroy[lane_inds]
    out_img = np.dstack((binary_img, binary_img, binary_img))
    return linex, liney, out_img, x_current, movepix


def fitPoly2(binary_warped_img):
    leftx, lefty, rightx, righty,out_img = findLanePixels2(binary_warped_img)
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    except TypeError:
        pass
    ploty = linspace(0, binary_warped_img.shape[0]-1, binary_warped_img.shape[0])
    
    try:
        leftx_fit = left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
        rightx_fit = right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]
    except TypeError:
        leftx_fit = ploty**2+ploty
        rightx_fit = ploty**2+ploty
    print(left_fit)
    
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx]= [0, 0, 255]
    mid_fit = [(left_fit[0]+right_fit[0])/2, (left_fit[1]+right_fit[1])/2, (left_fit[2]+right_fit[2])/2]
    midx_fit = mid_fit[0]*ploty**2 + mid_fit[1]*ploty + mid_fit[2]
    mid_curve = np.column_stack((midx_fit.astype(np.int32), ploty.astype(np.int32)))
    cv2.polylines(out_img, [mid_curve], False, (0,0,255))
    left_curve = np.column_stack((leftx_fit.astype(np.int32), ploty.astype(np.int32)))
    right_curve = np.column_stack((rightx_fit.astype(np.int32), ploty.astype(np.int32)))
    cv2.polylines(out_img, [left_curve, right_curve], False, (0,255,255))
    # plt.plot(leftx_fit, ploty, color='yellow')
    return mid_fit, out_img

def fitPoly1 (binary_img):
    linex, liney, out_img, x_current, movepix= findLanePixels1(binary_img)
    try:
        first_fit = np.polyfit( liney, linex, 2)
        second_fit = np.array([first_fit[0], first_fit[1], first_fit[2]+movepix])
    except TypeError:
        print('yoooooooooooooooooooo')
        first_fit = np.array([7,2,1])
        second_fit = np.array([7,2,1])
        pass

    ploty = linspace(0, binary_img.shape[0]-1, binary_img.shape[0])

    try:
        first_fitx = first_fit[0]*ploty**2 + first_fit[1]*ploty + first_fit[2]
        second_fitx = second_fit[0]*ploty**2 + second_fit[1]*ploty + second_fit[2]
    except TypeError:
        print('Could\'nt find the polynomial')
        first_fitx = 1*ploty**2 + 1*ploty
        second_fitx = 1*ploty**2 + 1*ploty
    print(first_fit)
    print(second_fit)
    out_img[liney, linex] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]
    mid_fit = [(first_fit[0]+second_fit[0])/2, (first_fit[1]+second_fit[1])/2, (first_fit[2]+second_fit[2])/2]
    mid_fitx = mid_fit[0]*ploty**2 + mid_fit[1]*ploty + mid_fit[2]
    mid_curve = np.column_stack((mid_fitx.astype(np.int32), ploty.astype(np.int32)))
    cv2.polylines(out_img, [mid_curve], False, (0,0,255))
    first_curve = np.column_stack((first_fitx.astype(np.int32), ploty.astype(np.int32)))
    second_curve = np.column_stack((second_fitx.astype(np.int32), ploty.astype(np.int32)))
    cv2.polylines(out_img, [first_curve, second_curve], False, (0,255,255))
    # cv2.imshow('out img', out_img)
    # cv2.waitKey(0)
    return mid_fit, out_img


def calculate_pd(warped_binary_img):
    global d, p, past_d, past_p
    laneNumber = oneOrTwo(warped_binary_img)
    if laneNumber ==1:
        mid_fit, out_img=fitPoly1(warped_binary_img)
    else:
        mid_fit, out_img = fitPoly2(warped_binary_img)
    midpoint_img = np.int32(warped_binary_img.shape[1]//2)
    midpoint_poly = np.polyval(mid_fit, warped_binary_img.shape[0]-8)
    p = midpoint_img - midpoint_poly
    x_prime = np.poly1d(mid_fit)
    mid_derivative_equ = x_prime.deriv()
    mid_derivative = mid_derivative_equ(warped_binary_img.shape[0]-1)
    d = np.arctan([mid_derivative])*180/np.pi
    print('d=', d)
    # print('d=', d)
    if (-70<p<70):
        past_p = p
    if (-13<d<13):
        past_d = d
    print('past d=', past_d)
    return past_p, past_d, out_img

def laneStatus(yellow_line_status, white_line_status):
    global lane_status
    if (yellow_line_status == 0):
        # if (white_line_status ==2):
        lane_status = 1
    elif (yellow_line_status == 1):
        # if (white_line_status ==2):
        lane_status = 2
    elif (yellow_line_status == 2):
        if (white_line_status == 0):
            lane_status = 0
        elif (white_line_status == 1):
            lane_status = 3
    return lane_status


#Calling the class
car = AVISEngine.car()

#connecting to the server (Simulator)
car.connect("127.0.0.1", 25001)

#Counter variable
counter = 0

debug_mode = False
#sleep for 3 seconds to make sure that client connected to the simulator 
time.sleep(3)
try:
    while(True):
        #Counting the loops
        
        counter = counter + 1

        #Set the power of the engine the car to 20, Negative number for reverse move, Range [-100,100]
        car.setSpeed(10)

        #Set the Steering of the car -10 degree from center
        car.setSteering(0)

        #Get the data. Need to call it every time getting image and sensor data
        car.getData()

        #Start getting image and sensor data after 4 loops. for unclear some reason it's really important 
        if(counter > 4):
            #returns a list with three items which the 1st one is Left sensor data, the 2nd one is the Middle Sensor data, and the 3rd is the Right one.
            sensors = car.getSensors() 
            #EX) sensors[0] returns an int for left sensor data in cm

            #returns an opencv image type array. if you use PIL you need to invert the color channels.
            image = car.getImage()
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #returns an integer which is the real time car speed in KMH
            warped_img = warp(rgb_image)
            binary_img = colorThreshold(warped_img)
            yellow_bmask_img = yellowDetection(warped_img)
            # binary2_img = canny(warped_img)
            # binary_img = combine(binary1_img, binary2_img)
            p, d, out_img= calculate_pd(binary_img)
            
            # yellowLineStatus = yellowLineLoc(yellow_bmask_img)
            # if ((p!=0) & (d!=0)):
            #     pre_p = p
            #     pre_d = d
            yellow_line_status, white_line_status = outOfRoadLane(binary_img, yellow_bmask_img)
            lane_status = laneStatus(yellow_line_status, white_line_status)


            carSpeed = car.getSpeed()
            # print('yellow line status=',yellow_line_status)
            # print('white status=', white_line_status)
            print('LANE STATUS', lane_status)

            #im a bitch she is a boss ^~~^
            #Don't print data for better performance
            if(debug_mode):
                print("Speed : ",carSpeed) 
                #currently the angle between the sensors is 30 degree TODO : be able to change that from conf.py
                print("Left : " + str(sensors[0]) + "   |   " + "Middle : " + str(sensors[1])  +"   |   " + "Right : " + str(sensors[2]))
            
            # cv2.imshow('warped img', warped_img)
            # cv2.imshow('binary1', binary1_img)

            #im a bitch she is a boss ^~~^
            #break the loop when q pressed
            # cv2.imshow('rec img', rec_img)
            cv2.imshow('yellow', yellow_bmask_img)
            cv2.imshow('binary', binary_img)
            cv2.imshow('output img', out_img)
            # cv2.imshow('canny img', binary2_img)
            #showing the opencv type image
            cv2.imshow('frames', image)
            #break the loop when q pressed
            # findLanePixels(binary_img)
            if cv2.waitKey(10) == ord('q'):
                break
            time.sleep(0.001)
        #A brief sleep to make sure everything 
        
finally:
    car.stop()
