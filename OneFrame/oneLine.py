import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.lib.polynomial import polyval
import math


def warp(rgb_image):
    pts1 = np.float32([[65,145], [205,145], [0,195], [255,195]])
    pts2 = np.float32([[0,0], [250,0], [0,350], [250,350]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped_image = cv2.warpPerspective(rgb_image, matrix, (250, 350))
    return warped_image


def colorThresh(warped_image):
    channel=warped_image[:, :, 0]
    thresh= (170,255)
    output=np.zeros_like(channel)
    output[(channel>= thresh[0]) & (channel<= thresh[1])]=255
    return output


def histogram(binary_img):
    image = binary_img//255
    bottom_half = image[image.shape[0]//2: , :]
    hist = np.sum(bottom_half, axis=0)
    return hist


def find_lane_pixels(binary_img):
    hist = histogram(binary_img)
    base = np.argmax(hist)
    midpoint = binary_img.shape[1]//2
    if (base<midpoint):
        movepix = np.int32(80)
    else:
        movepix = np.int32(-80)
    nwindows = 8
    margin = 20
    minpix = 30

    window_height = np.int32(binary_img.shape[0]//nwindows)
    x_current = base

    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    lane_inds = []
    rec = 1
    for window in range (nwindows):
        # print('base', leftx_base, rightx_base)
        # print('current', leftx_current, rightx_current)
        win_y_low = binary_img.shape[0] - ((window+1)*window_height)
        win_y_high = binary_img.shape[0] - ( window*window_height)
        win_x_low = x_current - margin
        win_x_high = x_current + margin

        # win_y_low = binary_img.shape[0] - ((window+1)*window_height)
        # win_y_high = binary_img.shape[0] - ( window*window_height)
        # if (leftx_current-margin<0):
        #     win_xleft_low=0
        # else:    
        #     win_xleft_low = leftx_current - margin
        # if (rightx_current+margin>binary_img.shape[1]):
        #     win_xright_high = binary_img.shape[1]
        # else:     
        #     win_xright_high = rightx_current + margin
        # win_xleft_high = leftx_current + margin
        # win_xright_low = rightx_current - margin

        # print('current left', leftx_current, 'current right', rightx_current)
        # print('left low:', win_xleft_low, 'left high', win_xleft_high)
        # print('right low:', win_xright_low,'right high',win_xright_high)
        
        if (rec != 0):
            cv2.rectangle( binary_img, (win_x_low , win_y_low), (win_x_high , win_y_high), (255,0,0), 2)
        # if (right_rec != 0):
        #     cv2.rectangle( binary_img, (win_xright_low , win_y_low), (win_xright_high , win_y_high), (255,0,0), 2)

        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        
        # if len(good_left_inds) > minpix:
        #     leftx_current = np.int( np.mean( nonzerox[good_left_inds]))
        #     left_lane_ind.append(good_left_inds)
        # else:
        #     break
        # if len(good_right_inds) > minpix:
        #     rightx_current = np.int( np.mean( nonzerox[good_right_inds]))
        #     right_lane_ind.append(good_right_inds)
        # else:
        #     break    

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
    # plt.imshow(binary_img)
    # plt.show()
    out_img = np.dstack((binary_img, binary_img, binary_img))
    return linex, liney, out_img, x_current, movepix
 

def fit_polynomial (binary_img):
    linex, liney, out_img, x_current, movepix= find_lane_pixels(binary_img)

    first_fit = np.polyfit( liney, linex, 2)
    second_fit = np.array([first_fit[0], first_fit[1], first_fit[2]+movepix])

    ploty = linspace(0, binary_img.shape[0]-1, binary_img.shape[0])

    try:
        first_fitx = first_fit[0]*ploty**2 + first_fit[1]*ploty + first_fit[2]
        second_fitx = second_fit[0]*ploty**2 + second_fit[1]*ploty + second_fit[2]
    except TypeError:
        print('Could\'nt find the polynomial')
        first_fitx = 1*ploty**2 + 1*ploty
        second_fitx = 1*ploty**2 + 1*ploty
    print(movepix)
    print(first_fit)
    print(second_fit)
    out_img[liney, linex] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    mid_fit = [(first_fit[0]+second_fit[0])/2, (first_fit[1]+second_fit[1])/2, (first_fit[2]+second_fit[2])/2]
    mid_fitx = mid_fit[0]*ploty**2 + mid_fit[1]*ploty + mid_fit[2]
    mid_curve = np.column_stack((mid_fitx.astype(np.int32), ploty.astype(np.int32)))
    cv2.polylines(out_img, [mid_curve], False, (0,0,255))
    first_curve = np.column_stack((first_fitx.astype(np.int32), ploty.astype(np.int32)))
    second_curve = np.column_stack((second_fitx.astype(np.int32), ploty.astype(np.int32)))
    cv2.polylines(out_img, [first_curve, second_curve], False, (0,255,255))
    cv2.imshow('out img', out_img)
    cv2.waitKey(0)
    # plt.plot(mid_fitx, ploty, color='red')
    # plt.imshow(out_img)
    # plt.show()
    return out_img, first_fitx, second_fitx ,mid_fit


image= cv2.imread('savedimg6.png')
image_copy= np.copy(image)
rgb_image= cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
warped_img = warp(rgb_image)
warped_binary_img = colorThresh(warped_img)
fit_polynomial(warped_binary_img)
