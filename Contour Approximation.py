# Import necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours as contours_package
import numpy as np
import imutils
import cv2
import math
import glob
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

def convert_to_slide_coordinate(pixel_image_coordinates, slide_corners):
    x_image_vector = [1, 0]
    y_image_vector = [0, 1]
    print("slide Corners:", slide_corners)
    #calculate normalized direction vectors for x and y for slide
    x_direction_vector = np.array([slide_corners[3][0]-slide_corners[0][0], slide_corners[3][1]-slide_corners[0][1]])
    x_norm = math.sqrt(x_direction_vector[0]**2 + x_direction_vector[1]**2)
    x_norm_direction = [x_direction_vector[0]/x_norm, x_direction_vector[1]/x_norm]
    y_direction_vector = np.array([slide_corners[2][0]-slide_corners[0][0], slide_corners[2][1]-slide_corners[0][1]])
    y_norm = math.sqrt(y_direction_vector[0]**2 + y_direction_vector[1]**2)
    y_norm_direction = [y_direction_vector[0]/y_norm, y_direction_vector[1]/y_norm]
    print("y_norm_direction:", y_norm_direction)
    # calculate theta (rotation angle from image coordinate to slide coordinate) using dot product of normalized vectors
    x_theta = math.acos((x_image_vector[0]*x_norm_direction[0]) + (x_image_vector[1]*x_norm_direction[1]))
    y_theta = math.acos((y_image_vector[0]*y_norm_direction[0]) + (y_image_vector[1]*y_norm_direction[1]))
    print("y_theta:", (y_image_vector[0]*y_norm_direction[0]) + (y_image_vector[1]*y_norm_direction[1]))
    theta = (x_theta+y_theta)/2
    print("theta:", theta)
    pixel_coordinates_matrix = np.asmatrix(pixel_image_coordinates)
    rows, _ = pixel_coordinates_matrix.shape
    # convert the matrix from 2D to 3D for transformation (z = 1)
    z_column = np.ones((rows, 1))
    pixel_coordinates_matrix = np.column_stack((pixel_coordinates_matrix, z_column))
    print("pixel_coordinates_matrix: \n", pixel_coordinates_matrix)
    transformation_matrix = np.array([[math.cos(theta), -1*math.sin(theta), -1*slide_corners[0][0]], [math.sin(theta), math.cos(theta), -1*slide_corners[0][1]], [0, 0, 1]])
    print("transformation Matrix: \n", transformation_matrix)
    pixel_slide_coordinates_3D = transformation_matrix.dot(pixel_coordinates_matrix.T).T
    print("pixel slide coordinates:", pixel_slide_coordinates_3D)
    # remove z column before returning pixel_slide_coordinates to make coordinates 2D
    pixel_slide_coordinates_2D = np.delete(pixel_slide_coordinates_3D, 2, axis=1)
    print("pixel slide coordinates:", pixel_slide_coordinates_2D)
    return pixel_slide_coordinates_2D


def imaging(input_contour_mask, input_tissue_height, input_tissue_width, pixelsPerMetric, step_size, x_top_left, y_top_left):
    # calculate the number of steps and the pixel spacing per step
    pixels_per_step = round(pixelsPerMetric*step_size)
    print("Pixels per step:", pixels_per_step)
    #num_step = round((input_tissue_height/pixelsPerMetric)/step_size)
    print("input tissue height: ", input_tissue_height)
    num_step = round(input_tissue_height/pixels_per_step)
    print("number of steps:", num_step)
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
    #cv2.imshow('mask', line_mask)
    cv2.imwrite('line_mask.jpg', line_mask)
    # use numpy.logical_and to determine the pixels where the lines intersect the contour mask
    imaging_rows = cv2.bitwise_and(line_mask, input_contour_mask)
    #imaging_rows = cv2.cvtColor(imaging_rows, cv2.COLOR_BGR2GRAY)
    print(imaging_rows.shape)
    #cv2.imshow('contour lines', imaging_rows)
    cv2.imwrite('contour_lines.jpg', imaging_rows)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    # find the pixel locations of the contour scanning pattern
    contour_pixel_lines = cv2.findNonZero(imaging_rows)

    # find the pixel locations of the grid scanning pattern
    grid_pixel_lines = cv2.findNonZero(line_mask)

    return np.vstack(contour_pixel_lines).squeeze(), np.vstack(grid_pixel_lines).squeeze()

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
            #cv2.imshow('slide',img)
            #cv2.waitKey()
            bounding_boxes.append([x, y, width, height])
    return bounding_boxes


def create_contour_mask(img, cnts):
    # create the mask image with all zeros
    mask = np.zeros(img.shape, dtype=np.float32)
    # draw the contour points onto the mask image & fill it in
    cv2.drawContours(mask, [cnts], 0, (255, 255, 255), -1)
    # find the points within the contour
    #pixelpoints = cv2.findNonZero(mask)
    #cv2.imshow('mask', mask)
    return mask

def find_corners(input_image, input_contours):
    height, width, _ = input_image.shape
    bw_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    image_area = height*width
    corners_image = input_image.copy()
    # search for the slide contour through the list of contours
    for contour in input_contours:
        size = cv2.contourArea(contour)
        # if the contour is of reasonable size (the slide contour) then create a mask
        if size > image_area/10:
            mask = create_contour_mask(bw_input_image, contour)
    #cv2.imshow('contours mask', mask)
    # find all non-zero points in the form [x,y] (column, row)
    slide_points = cv2.findNonZero(mask)
    # convert the points into a matrix to do matrix multiplication
    matrix_slide_pts = np.asmatrix(slide_points)
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
        cv2.circle(corners_image, (point[0], point[1]), 20, (0, 255, 255), -1)

    #cv2.imshow('mask w/ corners', input_image)
    cv2.imwrite('calculated_corners.jpg', corners_image)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    return np.asarray(points)

def find_tissue_contour(slide_width, slide_height, slide_threshold, pixels_per_micrometer):
    # calculate size of photo
    height, width = slide_threshold.shape
    print("Height of slide:", height)
    print("Width of slide:", width)
    # calculate the area of the slide
    slide_area = slide_width*slide_height
    # threshold the slide image
    tissue_contours, hierarchy = cv2.findContours(slide_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in tissue_contours:
        # check if contour is touching the border of the image (tissue will not be placed on edge of slide)
        bounding_x, bounding_y, bounding_width, bounding_height = cv2.boundingRect(contour)
        if bounding_x >= 0 and bounding_y >= 0 and bounding_x+bounding_width <= width-1 and bounding_y+bounding_height <= height-1:
            # check that the area of the contour is greater than 750000 micrometers and less than 375000000 micrometers
            if slide_area*0.004 < cv2.contourArea(contour) < slide_area*0.2:
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

# this function calculates the size of the slide in pixels
#
def calc_slide_size_pixels(corners):
    left_midpoint = midpoint(corners[0], corners[2]) # top left and bottom left midpoint
    right_midpoint = midpoint(corners[3], corners[1]) # top right and bottom right midpoint
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

def calc_cropped_corners(corner_points):
    top_x = np.amin(corner_points[:, 0])
    bottom_x = np.amax(corner_points[:, 0])
    left_y = np.amin(corner_points[:, 1])
    right_y = np.amax(corner_points[:, 1])
    top_left_cropped = np.asarray([corner_points[0][0]-top_x, corner_points[0][1]-left_y])
    bottom_right_cropped = np.asarray([top_left_cropped[0]+(corner_points[1][0]-corner_points[0][0]), top_left_cropped[1]+(corner_points[1][1]-corner_points[0][1])])
    bottom_left_cropped = np.asarray([top_left_cropped[0]+(corner_points[2][0]-corner_points[0][0]), top_left_cropped[1]+(corner_points[2][1]-corner_points[0][1])])
    top_right_cropped = np.asarray([top_left_cropped[0]+(corner_points[3][0]-corner_points[0][0]), top_left_cropped[1]+(corner_points[3][1]-corner_points[0][1])])

    cropped_corners = np.asarray([top_left_cropped, bottom_right_cropped, bottom_left_cropped, top_right_cropped])
    return cropped_corners

def scan_tissue(old_camera_matrix_file, camera_distortion_coeff_file, new_camera_matrix_file, image_file, step_size_microm, robot_resolution):
    # Calibrate Camera
    # orig_camera_matrix, distortion_coeff, new_camera_mtx = calibrate_camera()
    orig_camera_matrix = np.load(old_camera_matrix_file)
    distortion_coeff = np.load(camera_distortion_coeff_file)
    new_camera_mtx = np.load(new_camera_matrix_file)

    # Import image
    image = cv2.imread(image_file)
    # apply the calculated camera matrix and distortion coefficients to the image to undistort
    image = cv2.undistort(image, orig_camera_matrix, distortion_coeff, None, new_camera_mtx)
    # display and save undistorted image
    #cv2.imshow('undistorted',image)
    cv2.imwrite('undistorted.jpg',image)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    # Set the step size to 50 micrometer (0.05mm) (1000 um = 1mm)
    # step_size_microm = 500
    # Set the slide width to 25mm (25000 micrometer)
    slide_microm_width = 25000
    # set the slide height to 75mm (75000 micrometer)
    slide_microm_height = 75000

    # covert image to greyscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("image_black_and_white.jpg", img)
    #cv2.imshow('image', img)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    # calculate the threshold
    _, threshold = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite("image_threshold.jpg", threshold)

    #cv2.imshow('threshold',threshold)
    #cv2.imshow('inv_threshold',inv_threshold)

    # Find contour of the tissue slide
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours onto display image
    contours_copy = image.copy()

    # calculate the corners of the slide
    corners = find_corners(contours_copy, contours)
    # calculate the height and width of the slide in pixels
    slide_pixel_height, slide_pixel_width = calc_slide_size_pixels(corners)
    micrometer_per_pixel_width = slide_microm_width/slide_pixel_width
    micrometer_per_pixel_height = slide_microm_height/slide_pixel_height
    micrometer_per_pixel = (micrometer_per_pixel_width + micrometer_per_pixel_height)/2
    print("Micrometer per pixel:", micrometer_per_pixel)
    print("robot_resolution/micrometer_per_pixel", (micrometer_per_pixel/robot_resolution))
    scaling_factor = (micrometer_per_pixel/robot_resolution) * 2
    print("scaling_factor: ", scaling_factor)


    # crop the photo to size of the slide using the min and max x and y corner points
    crop_img = image[np.amin(corners[:,1]):np.amax(corners[:,1]), np.amin(corners[:,0]):np.amax(corners[:,0])]
    cv2.imwrite("cropped_image.jpg", crop_img)

    # increase contrast of image to brighten tissue sample
    crop_img_contrast = correct_contrast_brightness(crop_img, 2, 0)
    cv2.imwrite("cropped_contrast.jpg", crop_img_contrast)

    # convert image to greyscale
    grey_crop = cv2.cvtColor(crop_img_contrast, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('cropped image',grey_crop)
    cv2.imwrite('cropped_image_grey.jpg', grey_crop)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # threshold the image
    _, slide_threshold = cv2.threshold(grey_crop, 130, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)




    #cv2.drawContours(crop_img, contours, -1, (0, 0, 255), 2)
    cv2.imwrite("slide_threshold.jpg", slide_threshold)

    # upscale image for better mm to pixel accuracy
    slide_threshold = cv2.resize(slide_threshold, (0, 0), fx=scaling_factor, fy=scaling_factor)

    # upscale the corners points by the same scaling factor
    corners = corners*scaling_factor

    # convert corner points to fit in the cropped image
    corners = calc_cropped_corners(corners)

    # calculate the size of the slide in pixels (height and width)
    slide_pixel_height, slide_pixel_width = calc_slide_size_pixels(corners)

    # calculate the number of pixels per micrometer (average the pixels per micrometer height and width)
    pixels_per_micrometer_width = slide_pixel_width/slide_microm_width
    pixels_per_micrometer_height = slide_pixel_height/slide_microm_height
    pixels_per_micrometer = (pixels_per_micrometer_height + pixels_per_micrometer_width)/2
    print("Micrometer per pixel: ", ((slide_microm_height/slide_pixel_height)+(slide_microm_width/slide_pixel_width))/2)

    # find the contour of the tissue on the slide
    tissue_contour = find_tissue_contour(slide_pixel_width, slide_pixel_height, slide_threshold, pixels_per_micrometer)
    tissue_contour_image = cv2.resize(crop_img.copy(), (0, 0), fx=scaling_factor, fy=scaling_factor)
    cv2.drawContours(tissue_contour_image, tissue_contour, -1, (0, 255, 255), 5)
    cv2.imwrite("tissue_contour.jpg", tissue_contour_image)


    # generate a mask of the tissue contour
    tissue_mask = create_contour_mask(slide_threshold, tissue_contour)

    #cv2.imshow('tissue_mask', tissue_mask)
    cv2.imwrite('tissue_mask.jpg', tissue_mask)

    # calculate the size of the tissue in pixels
    tissue_width, tissue_height, top_left_pt = calc_tissue_size(tissue_contour)

    # calculate the linear scanning coordinates in pixels
    scan_pixel_img_coord, grid_pixel_img_coord = imaging(tissue_mask, tissue_height, tissue_width, pixels_per_micrometer, step_size_microm, top_left_pt[0], top_left_pt[1])

    # plot the grid coordinates in red and the scan_pixel coordinates in green for a visualization
    grid_vs_contour_lines = cv2.resize(crop_img.copy(), (0, 0), fx=scaling_factor, fy=scaling_factor)
    for grid_point in grid_pixel_img_coord:
        cv2.circle(grid_vs_contour_lines, tuple(grid_point), 1, (0,0,255))
    for contour_point in scan_pixel_img_coord:
        cv2.circle(grid_vs_contour_lines, tuple(contour_point), 1, (0, 255, 0))
    cv2.imwrite("resulting_points.jpg", grid_vs_contour_lines)
    # convert from image coordinate system to slide coordinate system
    scan_pixel_slide_coord = convert_to_slide_coordinate(scan_pixel_img_coord, corners)
    # convert from pixels to micrometers (divide the pixel coordinates matrix by pixels_per_micrometer conversion to find micrometers)
    scan_microm_slide_coord = np.divide(scan_pixel_slide_coord, pixels_per_micrometer)

    # calculate the size of scan pixels and grid pixels to calculate percentage change/difference
    contour_pixel_lines_rows = scan_pixel_img_coord.shape[0]
    grid_pixel_lines_rows = grid_pixel_img_coord.shape[0]

    # calculate the difference in size between grid scanning and contour scanning to show improvement
    percent_decrease = ((grid_pixel_lines_rows - contour_pixel_lines_rows)/grid_pixel_lines_rows) * 100

    return percent_decrease, grid_pixel_lines_rows, contour_pixel_lines_rows

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def gather_results():
    old_camera_matrix_file = "camera_matrix_orig.npy"
    camera_distortion_coeff_file = "camera_dist_coeff.npy"
    new_camera_matrix_file = "camera_matrix_new.npy"

    # Set the step size to 50 micrometer (0.05mm) (1000 um = 1mm)
    step_size_microm = 50

    # robot arm resolution set to 50 micrometer (0.05mm)
    robot_resolution = 50

    # initialize variables
    total_percent_decrease = 0
    total_num_photos = 0
    total_grid_pts = 0
    total_contour_pts = 0

    images = glob.glob('Tissue Images/*')
    for image_file in images:
        print("Image File:", image_file)
        total_num_photos += 1
        percent_decrease, num_grid_pts, num_contour_pts = scan_tissue(old_camera_matrix_file, camera_distortion_coeff_file, new_camera_matrix_file, image_file, step_size_microm, robot_resolution)
        total_percent_decrease += percent_decrease
        total_grid_pts += num_grid_pts
        total_contour_pts += num_contour_pts
    average_percent_change = total_percent_decrease/total_num_photos

    print("\n\n\n--------------------------------\n\n\n")
    print("Total number of photos:", total_num_photos)
    print("Average Percent Decrease:", average_percent_change)

def example_run():
    old_camera_matrix_file = "camera_matrix_orig.npy"
    camera_distortion_coeff_file = "camera_dist_coeff.npy"
    new_camera_matrix_file = "camera_matrix_new.npy"

    image_file = "Tissue Images/tissue20_2.jpg"

    # Set the step size to 50 micrometer (0.05mm) (1000 um = 1mm)
    step_size_microm = 100
    # set the robot resolution to 50 micrometer
    robot_resolution = 50

    _, _, _ = scan_tissue(old_camera_matrix_file, camera_distortion_coeff_file, new_camera_matrix_file, image_file, step_size_microm, robot_resolution)

example_run()

#gather_results()

