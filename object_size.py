# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2)

def object_size(image, cnts, width):

    # width of quarter (in inches)
    #width = 3.5

    # load the image, covert it to grayscale, blur it slightly
    #image = cv2.imread('slide.jpg')
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.medianBlur(gray,7)

    # invert the image color
    # inv_image = cv2.bitwise_not(gray)

    # perform edge detection, then perform a dilation + erosion to close gaps in b/w object edges
    # edged = cv2.Canny(inv_image, 50, 100)
    # edged = cv2.dilate(edged, None, iterations=1)
    # edged = cv2.erode(edged, None, iterations=1)

    # Calculate the threshold for the edges of the object in the image
    # _, threshold = cv2.threshold(inv_image, 20, 255, 0)

    # find contours in the edge map
    # cnts = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue

        #compute the rotated bounding box of the contour
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype='int')

        # order the points in the contour such that they appear in
        # top-left, top-right, bottom-right, bottom-left order
        # then draw the outline of the rotated bounding box
        box = perspective.order_points(box)
        #cv2.drawContours(image, [box.astype("int")], -1, (0,255,0), 2)

        # loop over the original points and draw them
        #for (x,y) in box:
        #    cv2.circle(image, (int(x), int(y)), 5, (0,0,255), -1)

        # unpack the ordered bounding box, then compute the midpoint b/w the top-left and top-right coordinates
        # then compute the midpoint b/w the bottom left and bottom right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint b/w the top-left and bottom-left points
        # then compute the midpoint b/w the top-right and bottom right points
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        #cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        #cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        #cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        #cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines b/w the midpoints
        #cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        #cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255,0,255), 2)

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

        # draw the object sizes on the image
        #cv2.putText(image, "{:.1f}mm".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        #cv2.putText(image, "{:.1f}mm".format(dimB), (int(trbrX - 15), int(trbrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

        # show the output image
        #cv2.imshow("Image", orig)
        #cv2.waitKey(0)
    return pixelsPerMetric
    #cv2.destroyAllWindows()