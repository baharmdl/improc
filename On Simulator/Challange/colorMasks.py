from numpy.core.function_base import linspace
import time
import cv2
import numpy as np


def yellowLightStatus(rgb_img):
    lower = np.uint8([180, 180,   0])
    upper = np.uint8([255, 255, 50])
    yellow_mask = cv2.inRange(rgb_img, lower, upper)
    nonzero_yellow = yellow_mask.nonzero()
    # print(len(nonzero_yellow[1]))
    if len(nonzero_yellow[1])>7000:
        yellow_light_status = 1
    else:
        yellow_light_status = 0
    return yellow_light_status


def redLightStatus(rgb_img):
    lower = np.uint8([150, 0, 0])
    upper = np.uint8([255, 50, 50])
    red_mask = cv2.inRange(rgb_img, lower, upper)
    nonzero_red = red_mask.nonzero()
    # print(len(nonzero_red[1]))
    if (len(nonzero_red[1])>7000) :
        red_light_status = 1
    else:
        red_light_status = 0
    return red_light_status, red_mask


def greenLightStatus(rgb_img):
    # print('green start')
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([80, 255, 50])
    green_mask = cv2.inRange(rgb_img, lower, upper)
    nonzero_green = green_mask.nonzero()
    # print(len(nonzero_green[1]))
    if (len(nonzero_green[1])>7000) :
        green_light_status = 1
    else:
        green_light_status = 0
        # print('wtffff')
    # print('green done')
    return green_light_status


def trafficLightStatus(rgb_img):
    green = greenLightStatus(rgb_img)
    yellow = yellowLightStatus(rgb_img)
    red, mask = redLightStatus(rgb_img)
    if green == 1:
        traffic_light_status = 1
        # print('yo1')
    elif yellow == 1:
        traffic_light_status = 2
        # print('yo2')
    elif red == 1:
        traffic_light_status = 3
        # print('yo3')
    else:
        # print('THIS SUCKS')
        traffic_light_status = 0
        # print('yo0')
    return traffic_light_status, mask


# cap = cv2.VideoCapture('test.mp4',cv2.CAP_V4L2)
# while(True):
#     ret, frame = cap.read()
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     cv2.imshow('frame',frame)
#     traffic_light_status, mask = trafficLightStatus(rgb_frame)
#     cv2.imshow('green', mask)
#     print(traffic_light_status)
#     if cv2.waitKey(15) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

vid_capture = cv2.VideoCapture('test.mp4')

if (vid_capture.isOpened() == False):
    print("Error opening the video file")
# Read fps and frame count
else:
# Get frame rate information
# You can replace 5 with CAP_PROP_FPS as well, they are enumerations
    fps = vid_capture.get(5)
    print('Frames per second : ', fps,'FPS')

# Get frame count
# You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
frame_count = vid_capture.get(7)
print('Frame count : ', frame_count)

while(vid_capture.isOpened()):
# vid_capture.read() methods returns a tuple, first element is a bool 
# and the second is frame
    ret, frame = vid_capture.read()
    if ret == True:
        cv2.imshow('Frame',frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        traffic_light_status, mask = trafficLightStatus(rgb_frame)
        print(traffic_light_status)
    # 20 is in milliseconds, try to increase the value, say 50 and observe
        key = cv2.waitKey(20)

        if key == ord('q'):
            break
    else:
        break

# Release the video capture object
vid_capture.release()
cv2.destroyAllWindows()



# img = cv2.imread('green.png')
# cv2.imshow('img',img)
# # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # traffic_light_status, mask = trafficLightStatus(rgb_img)
# # cv2.imshow('green', mask)
# cv2.waitKey(0)
# # print(traffic_light_status)