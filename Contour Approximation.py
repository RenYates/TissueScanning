import numpy as np
import cv2
from object_size import object_size
import random

# Import image
image = cv2.imread('slide.jpg')
cp_image = image.copy()

# Set the step size to 1mm and the slide size to 100mm
step_size = 2
slide_width = 100

# covert image to greyscale, blur it slightly, & threshold it
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img,7)
cv2.imshow('image',img)

# invert the image
inv_image = cv2.bitwise_not(img)
# calculate the threshold
_, threshold = cv2.threshold(inv_image, 20, 255, 0)

# Find contours
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
cv2.drawContours(cp_image, contours, -1, (0,0,0), 2)

pixelsPerMetric = object_size(image, contours, slide_width)

# stack the points and remove any unnecessary nesting
cnts = np.vstack(contours).squeeze()
#print cnts

# Find the bounding rectangle around the contour of the tissue slide
# first two values are the x and y coordinates of the top left of the box
# height and width are in pixels
x_top_left, y_top_left, width, height = cv2.boundingRect(cnts)
cv2.rectangle(image, (x_top_left,y_top_left), (x_top_left+width,y_top_left+height),(200,200,200),2)

# loop through and plot each contour point on the image (change the colour for visualization for order of point plots)
i = 0
for cont in cnts:
    cv2.circle(image, (int(cont[0]), int(cont[1])), 5, (0, 0, i), -1)
    i += 10

# convert height to mm & calculate the number of steps
# num_step = (height/pixelsPerMetric)/step_size
# pixels_per_step = pixelsPerMetric*step_size
# step = 0
# while step <= num_step:
#     # calculate the points for the horizontal line moving vertically down the image
#     horiz_left_point = (int(x_top_left), int(y_top_left + (step*pixels_per_step)))
#     horiz_right_point = (int(x_top_left + width), int(y_top_left + (step*pixels_per_step)))
#
#     # print the points onto the image & plot the line
#     cv2.circle(image, horiz_left_point, 5, (0,0,0), -1)
#     cv2.circle(image, horiz_right_point, 5, (0,0,0), -1)
#     cv2.line(image, horiz_left_point, horiz_right_point, (0,0,0), 2)
#
#     step += 1




cv2.imshow('Contours',image)
cv2.waitKey(0)
cv2.destroyAllWindows()