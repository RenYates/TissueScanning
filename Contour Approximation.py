# Import necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours as contours_package
import numpy as np
import imutils
import cv2
import math
from camera_calibration import calibrate_camera

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2)

def calc_dist_between_pts (ptA, ptB):
    x_diff = (ptB[0]-ptA[0])**2
    y_diff = (ptB[1]-ptA[1])**2
    return (math.sqrt(x_diff + y_diff))

def object_size(conts, width):

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (conts, _) = contours_package.sort_contours(conts)
    pixelsPerMetric = None

    # loop over the contours individually
    for c in conts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue

        #compute the rotated bounding box of the contour
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype='int')

        # order the points in the contour such that they appear in
        # top-left, top-right, bottom-right, bottom-left order
        box = perspective.order_points(box)

        # unpack the ordered bounding box, then compute the midpoint b/w the top-left and top-right coordinates
        # then compute the midpoint b/w the bottom left and bottom right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint b/w the top-left and bottom-left points
        # then compute the midpoint b/w the top-right and bottom right points
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # if the pixels per metric has not been initialized, then compute it as the ratio of pixels to supplied metric
        # in this case, mm
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / width

        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

    return pixelsPerMetric

# def convert_pixel_to_stage_coord(img, pixel_pts):
#     (x_max, y_max) = img.shape
#     x_midpoint = midpoint((0,0),(x_max,0))
#     y_midpoint = midpoint((0,0),(0,y_max))
#     center = (x_midpoint, y_midpoint)

def imaging(img, contour_mask, height, pixelsPerMetric, step_size, x_top_left, y_top_left):
    # convert height from pixels to micrometers & calculate the number of steps
    num_step = (height/pixelsPerMetric)/step_size
    pixels_per_step = pixelsPerMetric*step_size
    step = 0

    # create a grid mask image for the lines to be printed on
    line_mask = np.zeros(img.shape, np.uint8)

    # calculate and draw the horizontal lines onto the mask image
    while step <= num_step:
        # calculate the points for the horizontal line moving vertically down the image
        horiz_left_point = (int(x_top_left), int(y_top_left + (step*pixels_per_step)))
        horiz_right_point = (int(x_top_left + width), int(y_top_left + (step*pixels_per_step)))
#
        # print the points onto the image & plot the line
        #cv2.circle(line_mask, horiz_left_point, 5, (0,0,0), -1)
        #cv2.circle(line_mask, horiz_right_point, 5, (0,0,0), -1)
        cv2.line(line_mask, horiz_left_point, horiz_right_point, (255,255,255), 1)
        step += 1
    #cv2.imshow('mask', line_mask)
    # use numpy.logical_and to determine the pixels where the lines intersect the contour mask
    imaging_rows = cv2.bitwise_and(line_mask,contour_mask)

    # normalize the generated photo
    normalized_image = cv2.normalize(imaging_rows, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #print("Normalized Image pts:", normalized_image)
    #cv2.imshow('normalized_image', normalized_image)
    #cv2.imwrite('lines.png', normalized_image)
    pixel_lines = cv2.findNonZero(normalized_image)
    #print("normalized pixel_lines: ", pixel_lines)
    return pixel_lines

def find_slide_contour(img,cnts):
    bounding_boxes = []
    for cnt in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(cnt) < 100:
            continue
        # calculate the contour perimeter
        cnt_perimeter = cv2.arcLength(cnt,True)
        # calculate the approx polygon of the contour
        # 2nd argument is 4% of the cnt perimeter
        approx_polygon = cv2.approxPolyDP(cnt, 0.04 * cnt_perimeter, True)

        # detect if the shape is a rectangle or not
        if len(approx_polygon) == 4:
            #compute the bounding box of the contour and plot it
            (x, y, width, height) = cv2.boundingRect(approx_polygon)
            cv2.rectangle(img,(x,y), (x+width,y+height), (0,0,255), 3)
            cv2.imshow('slide',img)
            cv2.waitKey()
            bounding_boxes.append([x, y, width, height])
    return bounding_boxes

def create_contour_mask(img, cnts):
    # create the mask image with all zeros
    mask = np.zeros(img.shape, dtype=np.float32)
    # draw the contour points onto the mask image & fill it in
    cv2.drawContours(mask, [cnts], 0, (255, 255, 255), -1)
    # find the points within the contour
    #pixelpoints = cv2.findNonZero(mask)
    cv2.imshow('mask', mask)
    return mask

def find_corners(input_image, input_contours):
    # convert image to grayscale and change the type to float32
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    # search for the slide contour through the list of contours
    for contour in input_contours:
        size = cv2.contourArea(contour)
        # if the contour is of reasonable size (the slide contour) then create a mask
        if size > 1000:
            mask = create_contour_mask(gray, contour)
    cv2.imshow('contours mask', mask)
    # find all non-zero points in the form [x,y] (column, row)
    slide_points = cv2.findNonZero(mask)
    # convert the points into a matrix to do matrix multiplication
    matrix_slide_pts = np.asmatrix(slide_points)
    # 4 corners will be: min(x+y), min(-x-y), min(x-y), min(-x+y)
    # build an array with the correct + and - signs for multiplication
    multi_matrix = np.array([[1, -1, 1, -1], [1, -1, -1, 1]])
    # multiply the slide pts matrix to the multi_matrix
    result = matrix_slide_pts.dot(multi_matrix)
    print(result)

    # find the row index of the min value in each column in the result matrix
    # column 1 is top left, column 2 is bottom right, column 3 is bottom left, column 4 is top right
    indexes = []
    for column in range(result.shape[1]):
        indexes.append(np.argmin(result[:, column]))

    # find the corresponding (x,y) coordinates for each min value index
    points = []
    for index in indexes:
        points.append(slide_points[index][0])

    # plot the found points onto the mask image
    for point in points:
        cv2.circle(input_image, (point[0], point[1]), 5, (0, 255, 255), -1)

    cv2.imshow('mask w/ corners', input_image)
    cv2.imwrite('calculated_corners.jpg',input_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return np.asarray(points)

# Calibrate Camera
orig_camera_matrix, distortion_coeff, new_camera_mtx = calibrate_camera()

# Import image
image = cv2.imread('REIMS_slide_no_flash.jpg')
# resize the image smaller (for viewing)
image = cv2.resize(image, (0,0), fx=0.4, fy=0.4)
image = cv2.undistort(image, orig_camera_matrix, distortion_coeff, None, new_camera_mtx)
cv2.imshow('undistorted',image)
cv2.waitKey()
cv2.destroyAllWindows()
(max_y, max_x, _) = image.shape

# Set the step size to 50 micrometer (1000 um = 1mm)
step_size = 50
# Set the tissue width to 4mm (4000 micrometer)
tissue_width = 4000

# covert image to greyscale, blur it slightly, & threshold it
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#img = cv2.medianBlur(img,7)
#img = cv2.GaussianBlur(img, (5,5), 0)
#img = cv2.bilateralFilter(img,7,75,75)

cv2.imshow('image',img)
cv2.waitKey()
cv2.destroyAllWindows()


# invert the image
inv_image = cv2.bitwise_not(img)

# calculate the threshold
#threshold = cv2.adaptiveThreshold(img, 300, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11, 2)
#threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,31, 3)
_, threshold = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#threshold = cv2.bitwise_not(threshold)

cv2.imshow('threshold',threshold)
#cv2.imshow('inv_threshold',inv_threshold)

#create_hough_lines(image,edged)



# Find contour of the tissue slide
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# Draw contours onto display image
contours_copy = image.copy()
cv2.drawContours(contours_copy, contours, -1, (0,0,0), 2)

cv2.imshow('Contours',contours_copy)
cv2.waitKey()
cv2.destroyAllWindows()

corners_copy = image.copy()
corners = find_corners(corners_copy, contours)
print(corners)
cv2.imshow('original image', image)

# crop the photo to size of the slide using the min and max x and y corner points
crop_img = image[np.amin(corners[:,1]):np.amax(corners[:,1]), np.amin(corners[:,0]):np.amax(corners[:,0])]
grey_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

cv2.imshow('cropped image',crop_img)
cv2.waitKey()
cv2.destroyAllWindows()

# threshold and find contours of new cropped image to find contours of tissue
_, cropped_threshold = cv2.threshold(grey_crop, 130, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# find the contours of the image
#contours, hierarchy = cv2.findContours(cropped_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#lower_boundary = np.array([105,105,105], dtype='uint8')
#upper_boundary = np.array([192,192,192],dtype='uint8')

#mask = cv2.inRange(crop_img, lower_boundary, upper_boundary)
#output = cv2.bitwise_and(crop_img, crop_img, mask = mask)
#cv2.imshow("grey mask", np.hstack([crop_img, output]))
#cv2.waitKey()
#cv2.destroyAllWindows()

crop_img_copy = crop_img.copy()
cv2.drawContours(crop_img_copy, contours, -1, (0,0,0), 2)
cv2.imshow('cropped contours', crop_img_copy)

cv2.imshow('cropped', cropped_threshold)

# Find the bounding rectangle of each contour, find the largest bounding box to find the slide bounding box
# largest_area = 0
# for c in contours:
#     x,y,w,h = cv2.boundingRect(c)
#     # set a size parameter for the bounding rectangles, and prevent the image border contour from being selected
#     if w < (max_x/2) or w == max_x or h == max_y:
#         continue
#     area = w * h
#     if area > largest_area:
#         bounding_box = (x,y,w,h)
# plot the largest bounding box onto the image (aka the slide bounding box)
#(x,y,w,h) = bounding_box
#cv2.rectangle(copy_image, (x,y), (x+w, y+h),(0,255,0), 2)
#cv2.imshow('bounding rect', copy_image)
#cv2.waitKey()
#cv2.destroyAllWindows()

# rotated_rectangle = cv2.minAreaRect(true_corners)
# # box = cv2.boxPoints(rotated_rectangle)
# # (bottom_left, top_left, top_right, bottom_right) = box
# # #print("1:", bottom_left, "2:", top_left, "3:", top_right, "4:", bottom_right)
# # width = calc_dist_between_pts(top_left,top_right)
# # #print("width rotate:", width)
# # height = calc_dist_between_pts(top_left,bottom_left)
# # #print("height rotate:", height)
# # #if (width < max_x/2) or width == max_x or height == max_y:
# # #    continue
# # box = np.int0(box)
# # cv2.drawContours(copy_image, [box],0,(0,255,255),1)
# # cv2.imshow('bounding rect', copy_image)

# rotated bounding boxes around contour
# for c in contours:
#     rotated_rectangle = cv2.minAreaRect(c)
#     box = cv2.boxPoints(rotated_rectangle)
#     (bottom_left, top_left, top_right, bottom_right) = box
#     #print("1:", bottom_left, "2:", top_left, "3:", top_right, "4:", bottom_right)
#     width = calc_dist_between_pts(top_left,top_right)
#     #print("width rotate:", width)
#     height = calc_dist_between_pts(top_left,bottom_left)
#     #print("height rotate:", height)
#     if (width < max_x/2) or width == max_x or height == max_y:
#         continue
#     box = np.int0(box)
#     cv2.drawContours(copy_image, [box],0,(255,255,0),1)
# cv2.imshow('bounding rect', copy_image)

#cv2.imwrite('contours result.png',copy_image)


#for c in contours:
    # calculate accuracy as a percent of contour perimeter
#    accuracy = 0.03*cv2.arcLength(c,True)
#    approx = cv2.approxPolyDP(c, accuracy, True)
#    cv2.drawContours(image, [approx],0, (255,0,0), 2)
#    cv2.imshow('approx polyDP', image)


# Find the bounding box of the tissue slide
#slide_box = find_slide_contour(image,contours)

# pixelsPerMicrometer = object_size(contours, tissue_width)
#
# # stack the points and remove any unnecessary nesting
# cnts = np.vstack(contours).squeeze()
#
# # Create the mask for the contour & find pixel points (x,y) within contour
# contour_mask = create_contour_mask(img, cnts)
#
# # Find the bounding rectangle around the contour of the tissue slide
# # first two values are the x and y coordinates of the top left of the box
# # height and width are in pixels
# x_top_left, y_top_left, width, height = cv2.boundingRect(cnts)
# cv2.rectangle(image, (x_top_left,y_top_left), (x_top_left+width,y_top_left+height),(200,200,200),2)
#
# # calculate the pixel points of the horizontal scan lines on the tissue contour
# pixel_points = imaging(img, contour_mask, height, pixelsPerMicrometer, step_size, x_top_left, y_top_left)
#
# # normalize image coordinates (bottom left is (0,0), top right is (1,1)
# # normalized_pts = cv2.normalize(pixel_points, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)




cv2.waitKey(0)
cv2.destroyAllWindows()