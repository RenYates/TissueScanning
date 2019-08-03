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

#def convert_pixels_to_mm(pixel_coordinates, pixelsPerMetric, input_img, input_tissue_height, input_tissue_width):

def convert_slide_coordinate(pixel_image_coordinates, slide_corners):
    x_image_vector = [1, 0]
    y_image_vector = [0, 1]
    #calculate normalized direction vectors for x and y for slide
    x_direction_vector = np.array([slide_corners[3][0]-slide_corners[0][0], slide_corners[3][1]-slide_corners[0][1]])
    x_norm = math.sqrt(x_direction_vector[0]**2 + x_direction_vector[1]**2)
    x_norm_direction = [x_direction_vector[0]/x_norm, x_direction_vector[1]/x_norm]
    y_direction_vector = np.array([slide_corners[2][0]-slide_corners[0][0], slide_corners[2][1]-slide_corners[0][1]])
    y_norm = math.sqrt(y_direction_vector[0]**2 + y_direction_vector[1]**2)
    y_norm_direction = [y_direction_vector[0]/y_norm, y_direction_vector[1]/y_norm]
    print("y_norm_direction:", y_norm_direction)

    x_theta = math.acos((x_image_vector[0]*x_norm_direction[0]) + (x_image_vector[1]*x_norm_direction[1]))
    y_theta = math.acos((y_image_vector[0]*y_norm_direction[0]) + (y_image_vector[1]*y_norm_direction[1]))
    print("y_theta:", (y_image_vector[0]*y_norm_direction[0]) + (y_image_vector[1]*y_norm_direction[1]))
    #x_theta = math.asin(x_norm_direction[1]/x_norm_direction[0])
    #y_theta = math.asin(y_direction_vector[1]/y_direction_vector[0])
    theta = (x_theta+y_theta)/2
    print("theta:", theta)
    pixel_coordinates_matrix = np.asmatrix(pixel_image_coordinates)
    transformation_matrix = np.array([[math.cos(theta), -1*math.sin(theta), slide_corners[0][0]], [math.sin(theta), math.cos(theta), slide_corners[0][1]]])
    print("transformation Matrix: \n", transformation_matrix)
    pixel_slide_coordinates = pixel_coordinates_matrix.dot(transformation_matrix)
    print(x_norm_direction)
    print(y_norm_direction)
    print("pixel slide coordinates:", pixel_slide_coordinates)
    return pixel_slide_coordinates


def imaging(input_contour_mask, input_tissue_height, input_tissue_width, pixelsPerMetric, step_size, x_top_left, y_top_left):
    # convert height from pixels to micrometers & calculate the number of steps
    pixels_per_step = round(pixelsPerMetric*step_size)
    num_step = round((input_tissue_height/pixelsPerMetric)/step_size)
    print("pixels per step:", pixels_per_step)
    print("number of steps: ", num_step)
    step = 0

    # create a grid mask image for the lines to be printed on
    line_mask = np.zeros(input_contour_mask.shape, np.float32)

    # calculate and draw the horizontal lines onto the mask image
    while step <= num_step:
        # calculate the points for the horizontal line moving vertically down the image
        horiz_left_point = (int(x_top_left), int(y_top_left + (step*pixels_per_step)))
        horiz_right_point = (int(x_top_left + input_tissue_width), int(y_top_left + (step*pixels_per_step)))

        # print the points onto the image & plot the line
        #cv2.circle(line_mask, horiz_left_point, 5, (0,0,0), -1)
        #cv2.circle(line_mask, horiz_right_point, 5, (0,0,0), -1)
        cv2.line(line_mask, horiz_left_point, horiz_right_point, (255,255,255), 1, cv2.LINE_AA)
        step += 1
    cv2.imshow('mask', line_mask)
    cv2.imwrite('line_mask.jpg', line_mask)
    # use numpy.logical_and to determine the pixels where the lines intersect the contour mask
    imaging_rows = cv2.bitwise_and(line_mask, input_contour_mask)
    #imaging_rows = cv2.cvtColor(imaging_rows, cv2.COLOR_BGR2GRAY)
    print(imaging_rows.shape)
    #cv2.imshow('contour lines', imaging_rows)
    cv2.imwrite('contour_lines.jpg', imaging_rows)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # normalize the generated photo
    #normalized_image = cv2.normalize(imaging_rows, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #print("Normalized Image pts:", normalized_image)
    #cv2.imshow('normalized_image', normalized_image)
    #cv2.imwrite('lines.png', normalized_image)
    pixel_lines = cv2.findNonZero(imaging_rows)
    #print("normalized pixel_lines: ", pixel_lines)
    return np.vstack(pixel_lines).squeeze()

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
    #gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    #gray = np.float32(gray)
    height, width = input_image.shape
    image_area = height*width
    # search for the slide contour through the list of contours
    for contour in input_contours:
        size = cv2.contourArea(contour)
        # if the contour is of reasonable size (the slide contour) then create a mask
        if size > image_area/10:
            mask = create_contour_mask(input_image, contour)
    cv2.imshow('contours mask', mask)
    # find all non-zero points in the form [x,y] (column, row)
    slide_points = cv2.findNonZero(mask)
    # convert the points into a matrix to do matrix multiplication
    matrix_slide_pts = np.asmatrix(slide_points)
    print("Matrix_slide_pts: ", matrix_slide_pts)
    # 4 corners will be: min(x+y), min(-x-y), min(x-y), min(-x+y)
    # build an array with the correct + and - signs for multiplication
    multi_matrix = np.array([[1, -1, 1, -1], [1, -1, -1, 1]])
    # multiply the slide pts matrix to the multi_matrix
    result = matrix_slide_pts.dot(multi_matrix)

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

def find_tissue_contour(slide_width, slide_height, slide_threshold):
    # calculate the area of the slide
    slide_area = slide_width*slide_height
    # threshold the slide image
    tissue_contours, hierarchy = cv2.findContours(slide_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in tissue_contours:
        # check that the contour area is within this set boundary
        if slide_area * 0.004 < cv2.contourArea(contour) < slide_area * 0.3:
            #cv2.drawContours(crop_img_copy, contour, -1, (0, 255, 255), 2)
            #print(cv2.contourArea(contour))
            tissue_contour = contour
            return tissue_contour

def calc_tissue_size(tissue_contour):
    # calculate relative size of tissue contour using extreme pts and size of slide
    cnt_left_pt = tuple(tissue_contour[tissue_contour[:, :, 0].argmin()][0])
    cnt_right_pt = tuple(tissue_contour[tissue_contour[:, :, 0].argmax()][0])
    cnt_top_pt = tuple(tissue_contour[tissue_contour[:, :, 1].argmin()][0])
    cnt_bottom_pt = tuple(tissue_contour[tissue_contour[:, :, 1].argmax()][0])

    # calculate the width and height of the tissue sample using the edge points
    tissue_width = calc_dist_between_pts(cnt_left_pt, (cnt_right_pt[0],cnt_left_pt[1]))
    tissue_height = calc_dist_between_pts(cnt_top_pt, (cnt_top_pt[0],cnt_bottom_pt[1]))
    print(tissue_width)
    print(tissue_height)
    # plot the extreme points to validate their positioning
    #cv2.circle(input_image, cnt_left_pt, 5, (0, 0, 255), -1)
    #cv2.circle(input_image, (cnt_right_pt[0], cnt_left_pt[1]), 5, (255, 0, 0), -1)
    #cv2.circle(input_image, cnt_top_pt, 5, (0, 255, 0), -1)
    #cv2.circle(input_image, (cnt_top_pt[0], cnt_bottom_pt[1]), 5, (0, 255, 255), -1)
    #cv2.circle(input_image, (cnt_left_pt[0], cnt_top_pt[1]), 5, (255, 0, 255), -1)
    #cv2.imshow('plotted points', input_image)
    return tissue_width, tissue_height, (cnt_left_pt[0], cnt_top_pt[1])

def calc_slide_size_pixels(corners):
    left_midpoint = midpoint(corners[0],corners[2]) # top left and bottom left midpoint
    right_midpoint = midpoint(corners[3],corners[1]) # top right and bottom right midpoint
    top_midpoint = midpoint(corners[0], corners[3]) # top left and top right midpoint
    bottom_midpoint = midpoint(corners[2], corners[1]) # bottom left and bottom right midpoint
    slide_pixel_width = calc_dist_between_pts(left_midpoint, right_midpoint)
    slide_pixel_height = calc_dist_between_pts(top_midpoint, bottom_midpoint)
    return slide_pixel_height, slide_pixel_width

def correct_contrast_brightness(image, contrast, brightness):
    new_image = np.zeros(image.shape, image.dtype)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for color in range (image.shape[2]):
                new_image[y, x, color] = np.clip(contrast*image[y, x, color] + brightness, 0, 255)
    return new_image

def main():
    # Calibrate Camera
    # orig_camera_matrix, distortion_coeff, new_camera_mtx = calibrate_camera()
    orig_camera_matrix = np.load("camera_matrix_orig.npy")
    distortion_coeff = np.load("camera_dist_coeff.npy")
    new_camera_mtx = np.load("camera_matrix_new.npy")

    # Import image
    image = cv2.imread('Tissue Images/tissue20_4.jpg')
    # apply the calculated camera matrix and distortion coefficients to the image to undistort
    image = cv2.undistort(image, orig_camera_matrix, distortion_coeff, None, new_camera_mtx)
    # display and save undistorted image
    cv2.imshow('undistorted',image)
    cv2.imwrite('undistorted.jpg',image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    (max_y, max_x, _) = image.shape

    # Set the step size to 50 micrometer (0.05mm) (1000 um = 1mm)
    step_size_microm = 50
    # Set the slide width to 25mm (25000 micrometer)
    slide_microm_width = 25000
    # set the slide height to 75mm (75000 micrometer)
    slide_microm_height = 75000
    # set scaling factor for image
    scaling_factor = 4

    # covert image to greyscale, blur it slightly, & threshold it
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #img = cv2.medianBlur(img,7)
    #img = cv2.GaussianBlur(img, (5,5), 0)
    #img = cv2.bilateralFilter(img,7,75,75)

    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


    # invert the image
    #inv_image = cv2.bitwise_not(img)

    # calculate the threshold
    #threshold = cv2.adaptiveThreshold(img, 300, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11, 2)
    #threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,31, 3)
    _, threshold = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #threshold = cv2.bitwise_not(threshold)

    #cv2.imshow('threshold',threshold)
    #cv2.imshow('inv_threshold',inv_threshold)

    # Find contour of the tissue slide
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours onto display image
    #contours_copy = image.copy()
    #cv2.drawContours(contours_copy, contours, -1, (0,0,0), 2)

    #cv2.imshow('Contours',contours_copy)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    # set the fraction for slide size to image size to 10
    corners = find_corners(threshold, contours)
    print(corners)


    # # number of pixels in 25mm (25000 micrometers)
    # print("Slide Pixel Width: ", slide_pixel_width)
    # print("Slide Pixel Height: ", slide_pixel_height)
    # # resize the contour mask of the tissue using these values
    # width_resize = slide_microm_width/slide_pixel_width
    # print("Width Resize: ", width_resize)
    # height_resize = slide_microm_height/slide_pixel_height
    # print("Height_Resize: ", height_resize)
    # pixelsPerMicrometer = slide_pixel_width/slide_microm_width
    # print("Pixel Per Micrometer:", pixelsPerMicrometer)

    cv2.imshow('original image', image)

    # crop the photo to size of the slide using the min and max x and y corner points
    crop_img = image[np.amin(corners[:,1]):np.amax(corners[:,1]), np.amin(corners[:,0]):np.amax(corners[:,0])]
    # increase contrast of image to brighten tissue sample
    crop_img = correct_contrast_brightness(crop_img, 2, 0)
    # #height, width, _ = crop_img.shape
    grey_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    crop_height, crop_width = grey_crop.shape

    # cv2.imshow('cropped image',grey_crop)
    # cv2.imwrite('cropped_photo.jpg', crop_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # threshold and find contours of new cropped image to find contours of tissue
    _, slide_threshold = cv2.threshold(grey_crop, 130, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(slide_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(crop_img, contours, -1, (0, 0, 255), 2)
    cv2.imwrite("slide_threshold.jpg", slide_threshold)

    # upscale image for better mm to pixel accuracy
    slide_threshold = cv2.resize(slide_threshold, (0, 0), fx=scaling_factor, fy=scaling_factor)
    # upscale the corners points by the same scaling factor
    corners = corners*scaling_factor
    # calculate the pixelsPerMetric value using the corners of the slide
    slide_pixel_height, slide_pixel_width = calc_slide_size_pixels(corners)

    tissue_contour = find_tissue_contour(slide_pixel_width, slide_pixel_height, slide_threshold)
    # # find the contours of the image
    # cropped_contours, hierarchy = cv2.findContours(cropped_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #
    # #lower_boundary = np.array([105,105,105], dtype='uint8')
    # #upper_boundary = np.array([192,192,192],dtype='uint8')
    #
    # #mask = cv2.inRange(crop_img, lower_boundary, upper_boundary)
    # #output = cv2.bitwise_and(crop_img, crop_img, mask = mask)
    # #cv2.imshow("grey mask", np.hstack([crop_img, output]))
    # #cv2.waitKey()
    # #cv2.destroyAllWindows()
    #
    # image_area = height*width
    # crop_img_copy = crop_img.copy()
    # #crop_img_copy = cv2.resize(crop_img_copy, (0, 0), fx=4, fy=4)
    # for contour in cropped_contours:
    #     if image_area*0.006 < cv2.contourArea(contour) < image_area*0.3:
    #         cv2.drawContours(crop_img_copy, contour, -1, (0,255,255), 2)
    #         print(cv2.contourArea(contour))
    #         tissue_contour = contour
    # #print(tissue_contour)

    tissue_mask = create_contour_mask(slide_threshold, tissue_contour)
    # calculate the number of pixels per micrometer (average the pixels per micrometer height and width)
    pixelsPerMicrometerWidth = slide_pixel_width/slide_microm_width
    print("Pixels per Micrometer width: ", pixelsPerMicrometerWidth)
    pixelsPerMicrometerHeight = slide_pixel_height/slide_microm_height
    print("Pixels per Micrometer height: ", pixelsPerMicrometerHeight)
    pixelsPerMicrometer = (pixelsPerMicrometerHeight+pixelsPerMicrometerWidth)/2
    print("Pixels Per Micrometer: ", pixelsPerMicrometer)

    cv2.imshow('tissue_mask', tissue_mask)
    cv2.imwrite('tissue_mask.jpg', tissue_mask)
    tissue_width, tissue_height, top_left_pt = calc_tissue_size(tissue_contour)
    scan_pixel_img_coord = imaging(tissue_mask, tissue_height, tissue_width, pixelsPerMicrometer, step_size_microm, top_left_pt[0], top_left_pt[1])
    # convert from image coordinate system to slide coordinate system
    scan_pixel_slide_coord = convert_slide_coordinate(scan_pixel_img_coord, corners)
    # convert from pixels to micrometers




    #cv2.drawContours(crop_img_copy, cropped_contours, -1, (0,0,0), 2)
    #cv2.imshow('cropped contours', crop_img_copy)
    #cv2.imwrite('tissue_contour_found.jpg', crop_img_copy)

    #cv2.imshow('cropped', cropped_threshold)

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

main()


