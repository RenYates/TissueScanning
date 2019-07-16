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
    print("Normalized Image pts:", normalized_image)
    #cv2.imshow('normalized_image', normalized_image)
    #cv2.imwrite('lines.png', normalized_image)
    pixel_lines = cv2.findNonZero(normalized_image)
    print("normalized pixel_lines: ", pixel_lines)
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

def create_hough_lines(img, edges):

    maxLineGap = 100
    minLineLength = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('edges', edges)
    cv2.imshow('lines', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def create_contour_mask(img, cnts):
    # create the mask image with all zeros
    mask = np.zeros(img.shape, np.uint8)
    # draw the contour points onto the mask image & fill it in
    cv2.drawContours(mask, [cnts], 0, (255, 255, 255), -1)
    # find the points within the contour
    pixelpoints = cv2.findNonZero(mask)
    cv2.imshow('mask', mask)
    return mask

def harris_corners_detection(input_image, input_contours):
    for contour in input_contours:
        size = cv2.contourArea(contour)
        rect = cv2.minAreaRect(contour)
        if size > 1000:
            gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            mask = np.zeros(gray.shape, dtype=np.float32)
            cv2.fillPoly(mask, [contour], (255,255,255))
            dst = cv2.cornerHarris(mask, 5, 3, 0.04)
            ret, dst = cv2.threshold(dst, 0.1*dst.max(), 255, 0)
            dst = np.uint8(dst)
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners = cv2.cornerSubPix(gray, np.float32(centroids), (5,5), (-1,-1), criteria)
            input_image[dst>0.1*dst.max()] = [0,255,0]
            cv2.imshow('image',input_image)
            cv2.waitKey()
            cv2.destroyAllWindows()

def good_features_corner_detection(input_image, input_contours):
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    mask = np.ones(gray.shape, dtype=np.float32)
    for contour in input_contours:
            cv2.fillPoly(mask, [contour], [0,0,0])
    cv2.imshow('mask', mask)
    corners = cv2.goodFeaturesToTrack(mask, 10, 0.01, 10)
    corners = np.int0(corners)

    for corner in corners:
        x, y = corner.ravel()
        print("x:", x, "y:", y)
        cv2.circle(input_image, (x,y), 5, (0,255,0), -1)

    cv2.imshow('good features corners',input_image)
    cv2.imwrite('good_features_corners.png',input_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


# Calibrate Camera
#orig_camera_matrix, distortion_coeff, new_camera_mtx = calibrate_camera()

# Import image
image = cv2.imread('tissue_slide_black_bg.jpg')
#image = cv2.undistort(image, orig_camera_matrix, distortion_coeff, new_camera_mtx)
# resize the image smaller (for viewing)
image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
copy_image = image.copy()
good_features_image = image.copy()
harris_image = image.copy()
cv2.waitKey()
cv2.destroyAllWindows()
(max_y, max_x, _) = image.shape
print ("max x:", max_x,"max y:", max_y)

# Set the step size to 50 micrometer (1000 um = 1mm)
step_size = 30
# Set the tissue width to 4mm (4000 micrometer)
tissue_width = 3000

# covert image to greyscale, blur it slightly, & threshold it
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#img = cv2.medianBlur(img,7)
img = cv2.GaussianBlur(img, (5,5), 0)
#img = cv2.bilateralFilter(img,7,75,75)
cv2.imshow('image',img)

# invert the image
inv_image = cv2.bitwise_not(img)

# calculate the threshold
#threshold = cv2.adaptiveThreshold(img, 300, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11, 2)
#threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,31, 3)
_, threshold = cv2.threshold(img, 1, 300, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#inv_threshold = cv2.bitwise_not(threshold)

#cv2.imshow('threshold',threshold)
#cv2.imshow('inv_threshold',inv_threshold)

#create_hough_lines(image,edged)



# Find contour of the tissue slide
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

good_features_corner_detection(good_features_image, contours)
harris_corners_detection(harris_image, contours)

# Draw contours onto display image
cv2.drawContours(image, contours, -1, (0,0,0), 2)

cv2.imshow('Contours',image)

# Find the bounding rectangle of each contour, find the largest bounding box to find the slide bounding box
largest_area = 0
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    # set a size parameter for the bounding rectangles, and prevent the image border contour from being selected
    if w < (max_x/2) or w == max_x or h == max_y:
        continue
    area = w * h
    if area > largest_area:
        bounding_box = (x,y,w,h)
# plot the largest bounding box onto the image (aka the slide bounding box)
(x,y,w,h) = bounding_box
cv2.rectangle(copy_image, (x,y), (x+w, y+h),(0,255,0), 2)
cv2.imshow('bounding rect', copy_image)
cv2.waitKey()
cv2.destroyAllWindows()

# rotated bounding boxes
for c in contours:
    rotated_rectangle = cv2.minAreaRect(c)
    box = cv2.boxPoints(rotated_rectangle)
    (bottom_left, top_left, top_right, bottom_right) = box
    #print("1:", bottom_left, "2:", top_left, "3:", top_right, "4:", bottom_right)
    width = calc_dist_between_pts(top_left,top_right)
    #print("width rotate:", width)
    height = calc_dist_between_pts(top_left,bottom_left)
    #print("height rotate:", height)
    if (width < max_x/2) or width == max_x or height == max_y:
        continue
    box = np.int0(box)
    print(box)
    cv2.drawContours(copy_image, [box],0,(0,0,255),2)
    cv2.imshow('bounding rect', copy_image)

cv2.imwrite('contours result.png',copy_image)


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