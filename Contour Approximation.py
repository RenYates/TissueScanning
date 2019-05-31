# Import necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours as contours_package
import numpy as np
import imutils
import cv2


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2)

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

def convert_pixel_to_stage_coord(img, pixel_pts):
    (x_max, y_max) = img.shape
    x_midpoint = midpoint((0,0),(x_max,0))
    y_midpoint = midpoint((0,0),(0,y_max))
    center = (x_midpoint, y_midpoint)

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
    cv2.imshow('mask', line_mask)
    # use numpy.logical_and to determine the pixels where the lines intersect the contour mask
    imaging_rows = cv2.bitwise_and(line_mask,contour_mask)
    cv2.imshow('intersect', imaging_rows)
    cv2.imwrite('lines.png', imaging_rows)
    pixel_lines = cv2.findNonZero(imaging_rows)
    return pixel_lines


def create_contour_mask(img, cnts):
    # create the mask image with all zeros
    mask = np.zeros(img.shape, np.uint8)
    # draw the contour points onto the mask image & fill it in
    cv2.drawContours(mask, [cnts], 0, (255, 255, 255), -1)
    # find the points within the contour
    pixelpoints = cv2.findNonZero(mask)
    pixelpoints = np.vstack(pixelpoints).squeeze()
    cv2.imshow('mask', mask)
    return mask

# Import image
image = cv2.imread('slide.jpg')
cp_image = image.copy()

# Set the step size to 50 micrometer (1000 um = 1mm)
step_size = 200
# Set the tissue width to 4mm (4000 micrometer)
tissue_width = 4000

# covert image to greyscale, blur it slightly, & threshold it
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img,7)
cv2.imshow('image',img)

# invert the image
inv_image = cv2.bitwise_not(img)
# calculate the threshold
_, threshold = cv2.threshold(inv_image, 20, 255, 0)

# Find contour of the tissue slide
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
cv2.drawContours(image, contours, -1, (0,0,0), 2)

pixelsPerMicrometer = object_size(contours, tissue_width)

# stack the points and remove any unnecessary nesting
cnts = np.vstack(contours).squeeze()
print(contours)


# Create the mask for the contour & find pixel points (x,y) within contour
contour_mask = create_contour_mask(img, cnts)

# Find the bounding rectangle around the contour of the tissue slide
# first two values are the x and y coordinates of the top left of the box
# height and width are in pixels
x_top_left, y_top_left, width, height = cv2.boundingRect(cnts)
cv2.rectangle(image, (x_top_left,y_top_left), (x_top_left+width,y_top_left+height),(200,200,200),2)

# calculate the pixel points of the horizontal scan lines on the tissue contour
pixel_points = imaging(img, contour_mask, height, pixelsPerMicrometer, step_size, x_top_left, y_top_left)

# normalize image coordinates (bottom left is (0,0), top right is (1,1)


cv2.imshow('Contours',image)
cv2.waitKey(0)
cv2.destroyAllWindows()