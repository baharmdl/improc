import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.lib.polynomial import polyval
import math

image= cv2.imread('savedimg2.png')
image_copy= np.copy(image)
rgb_image= cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)


def warp(rgb_image):
    # pts1 = np.float32([[525, 325], [625, 325], [250, 700], [1080, 700]])
    # pts2 = np.float32([[0, 0], [500, 0], [0, 400], [500, 400]])
    # pts1 = np.float32([[100,100], [165,100], [15,200], [250,200]])
    # pts2 = np.float32([[0,0], [250,0], [0,350], [250,350]])
    pts1 = np.float32([[65,145], [205,145], [0,195], [255,195]])
    pts2 = np.float32([[0,0], [250,0], [0,350], [250,350]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped_image = cv2.warpPerspective(rgb_image, matrix, (250, 350))
    # plt.imshow(warped_image)
    # plt.show()
    return warped_image


def colorThresh(warped_image):
    channel=warped_image[:, :, 0]
    thresh= (170,255)
    output=np.zeros_like(channel)
    output[(channel>= thresh[0]) & (channel<= thresh[1])]=255
    # threshold_binary_img= cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    # plt.imshow(output, cmap='gray')
    # plt.show()
    return output


def canny(warped_image):
    gray = cv2.cvtColor(warped_image,cv2.COLOR_RGB2GRAY)
    blurred_img = cv2.GaussianBlur(gray, (5,5), 0)
    canny_binary_img = cv2.Canny(blurred_img, 50, 150)
    # plt.imshow(canny_binary_img , cmap='gray')
    # plt.show()
    return canny_binary_img


def combine(threshold_binary_img, canny_binary_img):
    combined_binary = np.zeros_like(threshold_binary_img)
    combined_binary[ (threshold_binary_img==255) | (canny_binary_img==255) ] =255
    # plt.imshow(canny_binary_img , cmap='gray')
    # plt.show()
    return combined_binary


def hist(binary_img):
    image = binary_img//255
    bottom_half = image[image.shape[0]//2: , :]
    hist = np.sum(bottom_half, axis=0)
    return hist


def find_lane_pixels(binary_img):
    histogram = hist(binary_img)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint


    nwindows = 8
    margin = 30
    minpix = 30

    window_height = np.int(combined_binary_img.shape[0]//nwindows)
    leftx_current = leftx_base
    rightx_current = rightx_base
    # print('base', leftx_base, rightx_base)

    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    right_lane_ind = []
    left_lane_ind = []
    # win_y_low = combined_binary_img.shape[0] - 1
    # win_y_high = combined_binary_img.shape[0]
    # win_xleft_low = leftx_current - margin
    # win_xleft_high = leftx_current + margin
    # win_xright_low = rightx_current - margin
    # win_xright_high = rightx_current + margin
    # for i in range (0, len(nonzerox)):
    #     if (i >=leftx_current-margin) & (i <=leftx_current+margin):
    #         left_lane_ind_x.append(i)
    # # for i in range (0, len(nonzeroy)):
    #     if (i >=combined_binary_img.shape[0]-window_height) & (i <=combined_binary_img.shape[0]):
    #         left_lane_ind_y.append(i)
    # left_lane_ind_x = nonzerox[ (nonzerox >= leftx_current-margin) & (nonzerox <= leftx_current+margin)]
    # left_lane_ind_y = nonzeroy[ (nonzeroy >= combined_binary_img.shape[0]-window_height) & (nonzeroy <= combined_binary_img.shape[0])]
    #left_lane_ind = nonzero_arr[ nonzerox[(nonzerox >= leftx_current-margin) : (nonzerox <= leftx_current+margin), None] , [None, nonzeroy[(nonzeroy >= combined_binary_img.shape[0]-window_height) : (nonzeroy <= combined_binary_img.shape[0])]]
    #left_lane_ind = nonzero_arr[ nonzerox[(leftx_current-margin) : (leftx_current+margin), None] , [None, nonzeroy[(combined_binary_img.shape[0]-window_height) : (combined_binary_img.shape[0])]]]
    #left_lane_ind = nonzero_arr[ (combined_binary_img.shape[0]-1-window_height) : (combined_binary_img.shape[0]-1)  , (leftx_current-margin) : (leftx_current+margin)]
    # left_lane_ind= nonzero_arr[ (nonzero_arr >= leftx_current-margin) & (nonzero_arr <= leftx_current+margin), :]
    left_rec, right_rec = 1, 1
    for window in range (nwindows):
        # print('base', leftx_base, rightx_base)
        # print('current', leftx_current, rightx_current)
        # win_y_low = binary_img.shape[0] - ((window+1)*window_height)
        # win_y_high = binary_img.shape[0] - ( window*window_height)
        # win_xleft_low = leftx_current - margin
        # win_xleft_high = leftx_current + margin
        # win_xright_low = rightx_current - margin
        # win_xright_high = rightx_current + margin


        win_y_low = binary_img.shape[0] - ((window+1)*window_height)
        win_y_high = binary_img.shape[0] - ( window*window_height)
        if (leftx_current-margin<0):
            win_xleft_low=0
        else:    
            win_xleft_low = leftx_current - margin
        if (rightx_current+margin>binary_img.shape[1]):
            win_xright_high = binary_img.shape[1]
        else:     
            win_xright_high = rightx_current + margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin

        # print('current left', leftx_current, 'current right', rightx_current)
        # print('left low:', win_xleft_low, 'left high', win_xleft_high)
        # print('right low:', win_xright_low,'right high',win_xright_high)
        
        if (left_rec != 0):
            cv2.rectangle( binary_img, (win_xleft_low , win_y_low), (win_xleft_high , win_y_high), (255,0,0), 2)
        if (right_rec != 0):
            cv2.rectangle( binary_img, (win_xright_low , win_y_low), (win_xright_high , win_y_high), (255,0,0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        
        
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

        if ((len(good_left_inds) > minpix) & (len(good_right_inds) > minpix)):
            leftx_current = np.int( np.mean( nonzerox[good_left_inds]))
            left_lane_ind.append(good_left_inds)
            rightx_current = np.int( np.mean( nonzerox[good_right_inds]))
            right_lane_ind.append(good_right_inds)
            # print('1')
        elif ((len(good_left_inds) > minpix) & (len(good_right_inds) < minpix)):  
            leftx_current = np.int( np.mean( nonzerox[good_left_inds]))
            left_lane_ind.append(good_left_inds)
            right_rec = 0
            # print('2')
        elif ((len(good_left_inds) < minpix) & (len(good_right_inds) > minpix)):   
            rightx_current = np.int( np.mean( nonzerox[good_right_inds]))
            right_lane_ind.append(good_right_inds) 
            left_rec = 0
            # print('3')

    try:
        right_lane_ind = np.concatenate(right_lane_ind)
        left_lane_ind = np.concatenate(left_lane_ind)
    except ValueError:
        pass
    
    leftx = nonzerox[left_lane_ind]
    rightx = nonzerox[right_lane_ind]
    lefty = nonzeroy[left_lane_ind]
    righty = nonzeroy[right_lane_ind]
    # plt.imshow(binary_img)
    # plt.show()
    out_img = np.dstack((binary_img, binary_img, binary_img))
    return leftx, rightx, lefty, righty, out_img, leftx_current, rightx_current


def fit_polynomial (binary_img):
    leftx, rightx, lefty, righty, out_img, leftx_current, rightx_current = find_lane_pixels(binary_img)

    left_fit = np.polyfit( lefty, leftx, 2)
    right_fit = np.polyfit( righty, rightx, 2)

    ploty = linspace(0, binary_img.shape[0]-1, binary_img.shape[0])
    print(left_fit)
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        print('Could\'nt find the polynomial')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    print(left_fit)
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    # print(left_fitx.size)
    mid_fit = [(left_fit[0]+right_fit[0])/2, (left_fit[1]+right_fit[1])/2, (left_fit[2]+right_fit[2])/2]
    mid_fitx = mid_fit[0]*ploty**2 + mid_fit[1]*ploty + mid_fit[2]
    mid_curve = np.column_stack((mid_fitx.astype(np.int32), ploty.astype(np.int32)))
    cv2.polylines(out_img, [mid_curve], False, (0,0,255))
    left_curve = np.column_stack((left_fitx.astype(np.int32), ploty.astype(np.int32)))
    right_curve = np.column_stack((right_fitx.astype(np.int32), ploty.astype(np.int32)))
    cv2.polylines(out_img, [left_curve, right_curve], False, (0,255,255))
    cv2.imshow('out img', out_img)
    plt.plot(mid_fitx, ploty, color='red')
    plt.imshow(out_img)
    plt.show()
    return out_img, left_fitx, right_fitx ,mid_fit

def calculate_pd (binary_img): 
    output_img, left_fitx, right_fitx, mid_fit= fit_polynomial(binary_img)
    midx_img = binary_img.shape[1]//2
    midx_road = np.polyval(mid_fit, binary_img.shape[0]-1//2)
    p = midx_img - midx_road
    print('p=', p)
    x_prime = np.poly1d(mid_fit)
    mid_derivative_equ = x_prime.deriv()
    mid_derivative = mid_derivative_equ(binary_img.shape[0]-1)
    # print('derivative=', mid_derivative)
    # d = mid_derivative(binary_img.shape[0]-1)*180/np.pi
    # print('d=', d)
    print('midline equ=',x_prime)
    print('midline deriv=', mid_derivative_equ)
    print('mid_der', mid_derivative,)
    # d = math.atan(mid_derivative)*180/np.pi
    d = np.arctan([mid_derivative])*180/np.pi
    print('d=', d)
    return p, d







warped_image = warp(rgb_image)
# cv2.imshow('warped', warped_image)
threshold_binary_img = colorThresh(warped_image)
cv2.imshow('yo',threshold_binary_img)
cv2.waitKey(0)
canny_binary_img = canny(warped_image)
combined_binary_img= combine(threshold_binary_img, canny_binary_img)
p, d = calculate_pd(combined_binary_img)
# output_img, left_fitx, right_fitx, mid_fit= fit_polynomial(combined_binary_img)
# midx_img = combined_binary_img.shape[1]/2
# midx_current = (leftx_current + rightx_current)/2
# p = midx_img - midx_current
# print(leftx_current, rightx_current, midx_current, midx_img)
# print(mid_fit)
# y = np.poly1d(mid_fit)
# derivative = y.deriv()
# print(derivative)
# d = derivative(midx_current)*180/np.pi
# print(d)
# print(p)
# midx_road = np.polyval(mid_fit, combined_binary_img.shape[0]-1)
# p = midx_img - midx_road
# print('p=', p)
# x_prime = np.poly1d(mid_fit)
# mid_derivative = x_prime.deriv()
# print('derivative=', mid_derivative)
# d = mid_derivative(combined_binary_img.shape[0]-1)*180/np.pi
# print('d=', d)

# margin = 100
# nonzero = combined_binary_img.nonzero()
# nonzerox = np.array(nonzero[0])
# rightx_low = right_fitx - margin
# rightx_high = right_fitx + margin
# for pix in range (0, len(right_fitx)):
#     x = ((nonzerox>=rightx_low[pix]) & (nonzerox<=rightx_high[pix]))
#     good_right_inds.append(x)
# print(good_right_inds.size)


# plt.imshow(output_img)
# plt.show()


