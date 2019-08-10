import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import cv2
import numpy as np
import cv2
import math
import time


#
# Tissue_Scanning_Module
#

class Tissue_Scanning_Module(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Tissue Scanning Module" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Tissue Scanning"]
    self.parent.dependencies = []
    self.parent.contributors = ["Lauren Yates (Queen's University)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is a model designed for automatic tissue contour detection and scanning pattern determination
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# Tissue_Scanning_ModuleWidget
#

class Tissue_Scanning_ModuleWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):

    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...
    self.logic = Tissue_Scanning_ModuleLogic()

    # Load widget from .ui file (created by Qt Designer)
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/Tissue_Scanning_Module.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    self.ui.cameraInputSelector.setMRMLScene(slicer.mrmlScene)
    #self.ui.outputSelector.setMRMLScene(slicer.mrmlScene)

    # set max and min values of step size microm
    self.ui.stepSizeNumSpinBox.setMinimum(1)
    self.ui.stepSizeNumSpinBox.setMaximum(10000)
    self.ui.stepSizeNumSpinBox.setValue(50)

    # connections
    self.ui.calibrateButton.connect('clicked(bool)', self.onCalibrateButton)
    self.ui.pictureButton.connect('clicked(bool)', self.onPictureButton)
    self.ui.contourButton.connect('clicked(bool)', self.onContourButton)
    self.ui.scanningButton.connect('clicked(bool)', self.onScanningButton)
    self.ui.cameraInputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onCameraSelect)
    #self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onCameraSelect()

  def cleanup(self):
    pass

  def onCalibrateButton(self):
    self.logic.calibrate_camera()

  def onCameraSelect(self):
    self.ui.pictureButton.enabled = self.ui.cameraInputSelector.currentNode()
    self.ui.calibrateButton.enabled = self.ui.cameraInputSelector.currentNode()
    self.logic.setCameraInputNode(self.ui.cameraInputSelector.currentNode())

  def onPictureButton(self):
    image = self.logic.takePicture()
    self.logic.setImage(image)
    print("Picture taken")
    self.ui.contourButton.enabled = self.ui.pictureButton.enabled
    #enableScreenshotsFlag = self.ui.enableScreenshotsFlagCheckBox.checked
    #imageThreshold = self.ui.imageThresholdSliderWidget.value
    #self.logic.run(self.ui.inputSelector.currentNode(), imageThreshold, enableScreenshotsFlag)

  def onContourButton(self):
    self.logic.determine_contour()
    self.ui.scanningButton.enabled = self.ui.contourButton.enabled

  def onScanningButton(self):
    self.logic.setStepSizeMicrom(self.ui.stepSizeNumSpinBox.value)
    self.logic.generate_scanning_pattern()

#
# Tissue_Scanning_ModuleLogic
#

class Tissue_Scanning_ModuleLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):

    self.cameraInputNode = None
    self.camera_matrix = []
    self.distortion_coeff = 0
    self.new_camera_mtx = []
    self.image = None
    self.tissue_contour = None
    self.tissue_mask = None
    # Set the slide width to 25mm (25000 micrometer)
    self.slide_microm_width = 25000
    # set the slide height to 75mm (75000 micrometer)
    self.slide_microm_height = 75000
    # set the step size to 500 micrometers
    self.step_size_microm = 500
    # set robot resolution to 50 micrometers
    self.robot_resolution = 50
    self.scaling_factor = 0
    self.pixels_per_micrometer = 0
    self.corners = []



  def setCameraInputNode(self, cameraInputNode):
      self.cameraInputNode = cameraInputNode

  def setImage(self, image):
    self.image = image

  def setStepSizeMicrom(self, step_size_microm):
    self.step_size_microm = step_size_microm

  def setCamera_Matrix(self, camera_matrix):
    self.camera_matrix = camera_matrix

  def setDistortion_Coeff(self, distortion_coeff):
    self.distortion_coeff = distortion_coeff

  def setNew_Camera_Matrix(self, new_camera_mtx):
    self.new_camera_mtx = new_camera_mtx

  def calibrate_camera(self):
    print("Beginning Camera Calibration\n")
    # set the required number of good photos taken
    required_num_photos = 15
    current_num_photos = 0

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), ... (6,5,0)
    object_points = np.zeros((7 * 7, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    obj_points = []  # 3d pt in real world space
    img_points = []  # 2d pts in image plane

    find_photos = True
    # take images
    print("INFO: Searching for chessboard")
    while find_photos:
      # take a picture of the current image
      img = self.takePicture()
      print("INFO: Image taken")
      # img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      # find the chess board corners
      ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

      # If found, add object points, image points (after refining them)
      if ret == True:
        print("INFO: Chessboard found, move chessboard to new position")
        current_num_photos += 1
        obj_points.append(object_points)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)

        time.sleep(3)
        print("INFO: Searching for chessboard")

      if current_num_photos == required_num_photos:
        find_photos = False

    print("Image collection completed, determining distortion and camera matrix")
    ret, camera_matrix, distortion_coeff, rotation_vec, translation_vec = cv2.calibrateCamera(obj_points, img_points,
                                                                                              gray.shape[::-1], None,
                                                                                              None)
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w, h), 1, (w, h))
    # undistort the image
    dst = cv2.undistort(img, camera_matrix, distortion_coeff, None, new_camera_mtx)

    # crop the image

    x, y, w, h = roi

    new_dst = dst[y:y + h, x:x + w]
    cv2.imwrite('calibration_result.png', dst)
    #np.save("camera_matrix_orig.npy", camera_matrix)
    self.setCamera_Matrix(camera_matrix)
    #np.save("camera_dist_coeff.npy", distortion_coeff)
    self.setDistortion_Coeff(distortion_coeff)
    np.save("camera_matrix_new.npy", new_camera_mtx)
    self.setNew_Camera_Matrix(new_camera_mtx)
    print("Calibration complete.")
    # return camera_matrix, distortion_coeff, new_camera_mtx

  def takePicture(self):
    # get the name of the selected camera node
    #name = self.cameraInputNode
    # take screenshot of image and read it as an array
    image = slicer.util.array("Image_Image")
    image = image[0]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

  def midpoint(self, ptA, ptB):
    return ((ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2)

  def calc_dist_between_pts(self, ptA, ptB):
    x_diff = (ptB[0] - ptA[0]) ** 2
    y_diff = (ptB[1] - ptA[1]) ** 2
    return (math.sqrt(x_diff + y_diff))

  def convert_to_slide_coordinate(self, pixel_image_coordinates, slide_corners):
    x_image_vector = [1, 0]
    y_image_vector = [0, 1]
    print("slide Corners:", slide_corners)
    # calculate normalized direction vectors for x and y for slide
    x_direction_vector = np.array(
      [slide_corners[3][0] - slide_corners[0][0], slide_corners[3][1] - slide_corners[0][1]])
    x_norm = math.sqrt(x_direction_vector[0] ** 2 + x_direction_vector[1] ** 2)
    x_norm_direction = [x_direction_vector[0] / x_norm, x_direction_vector[1] / x_norm]
    y_direction_vector = np.array(
      [slide_corners[2][0] - slide_corners[0][0], slide_corners[2][1] - slide_corners[0][1]])
    y_norm = math.sqrt(y_direction_vector[0] ** 2 + y_direction_vector[1] ** 2)
    y_norm_direction = [y_direction_vector[0] / y_norm, y_direction_vector[1] / y_norm]
    print("y_norm_direction:", y_norm_direction)
    # calculate theta (rotation angle from image coordinate to slide coordinate) using dot product of normalized vectors
    x_theta = math.acos((x_image_vector[0] * x_norm_direction[0]) + (x_image_vector[1] * x_norm_direction[1]))
    y_theta = math.acos((y_image_vector[0] * y_norm_direction[0]) + (y_image_vector[1] * y_norm_direction[1]))
    print("y_theta:", (y_image_vector[0] * y_norm_direction[0]) + (y_image_vector[1] * y_norm_direction[1]))
    theta = (x_theta + y_theta) / 2
    print("theta:", theta)
    pixel_coordinates_matrix = np.asmatrix(pixel_image_coordinates)
    rows, _ = pixel_coordinates_matrix.shape
    # convert the matrix from 2D to 3D for transformation (z = 1)
    z_column = np.ones((rows, 1))
    pixel_coordinates_matrix = np.column_stack((pixel_coordinates_matrix, z_column))
    print("pixel_coordinates_matrix: \n", pixel_coordinates_matrix)
    transformation_matrix = np.array([[math.cos(theta), -1 * math.sin(theta), -1 * slide_corners[0][0]],
                                      [math.sin(theta), math.cos(theta), -1 * slide_corners[0][1]], [0, 0, 1]])
    print("transformation Matrix: \n", transformation_matrix)
    pixel_slide_coordinates_3D = transformation_matrix.dot(pixel_coordinates_matrix.T).T
    print("pixel slide coordinates:", pixel_slide_coordinates_3D)
    # remove z column before returning pixel_slide_coordinates to make coordinates 2D
    pixel_slide_coordinates_2D = np.delete(pixel_slide_coordinates_3D, 2, axis=1)
    print("pixel slide coordinates:", pixel_slide_coordinates_2D)
    return pixel_slide_coordinates_2D

  def imaging(self, input_contour_mask, input_tissue_height, input_tissue_width, pixelsPerMetric, step_size, x_top_left,
              y_top_left):
    print("pixels per metric:", pixelsPerMetric)
    print("step size:", step_size)
    print("pixels per metric * step size", pixelsPerMetric*step_size)
    # calculate the number of steps and the pixel spacing per step
    pixels_per_step = round(pixelsPerMetric * step_size)
    print("Pixels per step:", pixels_per_step)
    # num_step = round((input_tissue_height/pixelsPerMetric)/step_size)
    print("input tissue height: ", input_tissue_height)
    num_step = round(input_tissue_height / pixels_per_step)
    print("number of steps:", num_step)
    step = 0

    # create a grid mask image for the lines to be printed on
    line_mask = np.zeros(input_contour_mask.shape, np.float32)

    # calculate and draw the horizontal lines onto the mask image
    while step <= num_step:
      # calculate the points for the horizontal line moving vertically down the image
      horiz_left_point = (int(x_top_left), int(y_top_left + (step * pixels_per_step)))
      horiz_right_point = (int(x_top_left + input_tissue_width), int(y_top_left + (step * pixels_per_step)))

      # print the points onto the image & plot the line
      # cv2.circle(line_mask, horiz_left_point, 5, (0,0,0), -1)
      # cv2.circle(line_mask, horiz_right_point, 5, (0,0,0), -1)
      cv2.line(line_mask, horiz_left_point, horiz_right_point, (255, 255, 255), 1, cv2.LINE_AA)
      step += 1
    # cv2.imshow('mask', line_mask)
    cv2.imwrite('C:/Users/15ly1/PycharmProjects/TissueScanning/line_mask.jpg', line_mask)
    # use numpy.logical_and to determine the pixels where the lines intersect the contour mask
    imaging_rows = cv2.bitwise_and(line_mask, input_contour_mask)
    # imaging_rows = cv2.cvtColor(imaging_rows, cv2.COLOR_BGR2GRAY)
    print(imaging_rows.shape)
    # cv2.imshow('contour lines', imaging_rows)
    cv2.imwrite('C:/Users/15ly1/PycharmProjects/TissueScanning/contour_lines.jpg', imaging_rows)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # find the pixel locations of the contour scanning pattern
    contour_pixel_lines = cv2.findNonZero(imaging_rows)

    # find the pixel locations of the grid scanning pattern
    grid_pixel_lines = cv2.findNonZero(line_mask)

    return np.vstack(contour_pixel_lines).squeeze(), np.vstack(grid_pixel_lines).squeeze()

  def find_slide_contour(self, img, cnts):
    bounding_boxes = []
    for cnt in cnts:
      # if the contour is not sufficiently large, ignore it
      if cv2.contourArea(cnt) < 100:
        continue
      # calculate the contour perimeter
      cnt_perimeter = cv2.arcLength(cnt, True)
      # calculate the approx polygon of the contour
      # 2nd argument is 4% of the cnt perimeter
      approx_polygon = cv2.approxPolyDP(cnt, 0.04 * cnt_perimeter, True)

      # detect if the shape is a rectangle or not
      if len(approx_polygon) == 4:
        # compute the bounding box of the contour and plot it
        (x, y, width, height) = cv2.boundingRect(approx_polygon)
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 3)
        # cv2.imshow('slide',img)
        # cv2.waitKey()
        bounding_boxes.append([x, y, width, height])
    return bounding_boxes

  def create_contour_mask(self, img, cnts):
    # create the mask image with all zeros
    mask = np.zeros(img.shape, dtype=np.float32)
    # draw the contour points onto the mask image & fill it in
    cv2.drawContours(mask, [cnts], 0, (255, 255, 255), -1)
    # find the points within the contour
    # pixelpoints = cv2.findNonZero(mask)
    # cv2.imshow('mask', mask)
    return mask

  def find_corners(self, input_image, input_contours):
    height, width, _ = input_image.shape
    bw_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    image_area = height * width
    corners_image = input_image.copy()
    # search for the slide contour through the list of contours
    for contour in input_contours:
      size = cv2.contourArea(contour)
      # if the contour is of reasonable size (the slide contour) then create a mask
      if size > image_area / 10:
        mask = self.create_contour_mask(bw_input_image, contour)
    # cv2.imshow('contours mask', mask)
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
      cv2.circle(corners_image, (point[0], point[1]), 5, (0, 255, 255), -1)

    # cv2.imshow('mask w/ corners', input_image)
    cv2.imwrite('C:/Users/15ly1/PycharmProjects/TissueScanning/calculated_corners.jpg', corners_image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return np.asarray(points)

  def find_tissue_contour(self, slide_width, slide_height, slide_threshold, pixels_per_micrometer):
    # calculate size of photo
    height, width = slide_threshold.shape
    print("Height of slide:", height)
    print("Width of slide:", width)
    # calculate the area of the slide
    slide_area = slide_width * slide_height
    # threshold the slide image
    tissue_contours, hierarchy = cv2.findContours(slide_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in tissue_contours:
      # check if contour is touching the border of the image (tissue will not be placed on edge of slide)
      bounding_x, bounding_y, bounding_width, bounding_height = cv2.boundingRect(contour)
      if bounding_x >= 0 and bounding_y >= 0 and bounding_x + bounding_width <= width - 1 and bounding_y + bounding_height <= height - 1:
        # check that the area of the contour is greater than 750000 micrometers and less than 375000000 micrometers
        if (slide_area * 0.004) / pixels_per_micrometer < cv2.contourArea(contour) / pixels_per_micrometer:
          print("Slide Area:", (slide_area * 0.004) / pixels_per_micrometer)
          print("contour area in micrometers:", cv2.contourArea(contour) / pixels_per_micrometer)
          # cv2.drawContours(crop_img_copy, contour, -1, (0, 255, 255), 2)
          # print(cv2.contourArea(contour))
          tissue_contour = contour
          return tissue_contour

  def calc_tissue_size(self, tissue_contour):
    # calculate relative size of tissue contour using extreme pts and size of slide
    cnt_left_pt = tuple(tissue_contour[tissue_contour[:, :, 0].argmin()][0])
    cnt_right_pt = tuple(tissue_contour[tissue_contour[:, :, 0].argmax()][0])
    cnt_top_pt = tuple(tissue_contour[tissue_contour[:, :, 1].argmin()][0])
    cnt_bottom_pt = tuple(tissue_contour[tissue_contour[:, :, 1].argmax()][0])

    # calculate the width and height of the tissue sample using the edge points
    tissue_width = self.calc_dist_between_pts(cnt_left_pt, (cnt_right_pt[0], cnt_left_pt[1]))
    tissue_height = self.calc_dist_between_pts(cnt_top_pt, (cnt_top_pt[0], cnt_bottom_pt[1]))
    print(tissue_width)
    print(tissue_height)
    # plot the extreme points to validate their positioning
    # cv2.circle(input_image, cnt_left_pt, 5, (0, 0, 255), -1)
    # cv2.circle(input_image, (cnt_right_pt[0], cnt_left_pt[1]), 5, (255, 0, 0), -1)
    # cv2.circle(input_image, cnt_top_pt, 5, (0, 255, 0), -1)
    # cv2.circle(input_image, (cnt_top_pt[0], cnt_bottom_pt[1]), 5, (0, 255, 255), -1)
    # cv2.circle(input_image, (cnt_left_pt[0], cnt_top_pt[1]), 5, (255, 0, 255), -1)
    # cv2.imshow('plotted points', input_image)
    return tissue_width, tissue_height, (cnt_left_pt[0], cnt_top_pt[1])

  def calc_slide_size_pixels(self, corners):
    left_midpoint = self.midpoint(corners[0], corners[2])  # top left and bottom left midpoint
    right_midpoint = self.midpoint(corners[3], corners[1])  # top right and bottom right midpoint
    top_midpoint = self.midpoint(corners[0], corners[3])  # top left and top right midpoint
    bottom_midpoint = self.midpoint(corners[2], corners[1])  # bottom left and bottom right midpoint
    slide_pixel_width = self.calc_dist_between_pts(left_midpoint, right_midpoint)
    slide_pixel_height = self.calc_dist_between_pts(top_midpoint, bottom_midpoint)
    return slide_pixel_height, slide_pixel_width

  def correct_contrast_brightness(self, image, contrast, brightness):
    new_image = np.zeros(image.shape, image.dtype)
    for y in range(image.shape[0]):
      for x in range(image.shape[1]):
        for color in range(image.shape[2]):
          new_image[y, x, color] = np.clip(contrast * image[y, x, color] + brightness, 0, 255)
    return new_image

  def calc_cropped_corners(self, corner_points):
    top_x = np.amin(corner_points[:, 0])
    bottom_x = np.amax(corner_points[:, 0])
    left_y = np.amin(corner_points[:, 1])
    right_y = np.amax(corner_points[:, 1])
    top_left_cropped = np.asarray([corner_points[0][0] - top_x, corner_points[0][1] - left_y])
    bottom_right_cropped = np.asarray([top_left_cropped[0] + (corner_points[1][0] - corner_points[0][0]),
                                       top_left_cropped[1] + (corner_points[1][1] - corner_points[0][1])])
    bottom_left_cropped = np.asarray([top_left_cropped[0] + (corner_points[2][0] - corner_points[0][0]),
                                      top_left_cropped[1] + (corner_points[2][1] - corner_points[0][1])])
    top_right_cropped = np.asarray([top_left_cropped[0] + (corner_points[3][0] - corner_points[0][0]),
                                    top_left_cropped[1] + (corner_points[3][1] - corner_points[0][1])])
    cropped_corners = np.asarray([top_left_cropped, bottom_right_cropped, bottom_left_cropped, top_right_cropped])
    return cropped_corners

  def determine_contour(self):
    print("contour determination started")

    # undistort image using camera calibration matrices
    image = cv2.undistort(self.image, self.camera_matrix, self.distortion_coeff, None, self.new_camera_mtx)

    # covert image to greyscale
    img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("C:/Users/15ly1/PycharmProjects/TissueScanning/image_black_and_white.jpg", img)
    # cv2.imshow('image', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # calculate the threshold
    _, threshold = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("C:/Users/15ly1/PycharmProjects/TissueScanning/image_threshold.jpg", threshold)

    # cv2.imshow('threshold',threshold)
    # cv2.imshow('inv_threshold',inv_threshold)

    # Find contour of the tissue slide
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours onto display image
    contours_copy = self.image.copy()

    # calculate the corners of the slide
    self.corners = self.find_corners(contours_copy, contours)
    # calculate the height and width of the slide in pixels
    slide_pixel_height, slide_pixel_width = self.calc_slide_size_pixels(self.corners)
    micrometer_per_pixel_width = self.slide_microm_width / slide_pixel_width
    micrometer_per_pixel_height = self.slide_microm_height / slide_pixel_height
    micrometer_per_pixel = (micrometer_per_pixel_width + micrometer_per_pixel_height) / 2
    print("Micrometer per pixel:", micrometer_per_pixel)
    print("robot_resolution/micrometer_per_pixel", (micrometer_per_pixel / self.robot_resolution))
    self.scaling_factor = (micrometer_per_pixel / self.robot_resolution) * 2
    print("scaling_factor: ", self.scaling_factor)

    # crop the photo to size of the slide using the min and max x and y corner points
    self.crop_img = self.image[np.amin(self.corners[:, 1]):np.amax(self.corners[:, 1]), np.amin(self.corners[:, 0]):np.amax(self.corners[:, 0])]
    cv2.imwrite("C:/Users/15ly1/PycharmProjects/TissueScanning/cropped_image.jpg", self.crop_img)

    # increase contrast of image to brighten tissue sample
    #crop_img_contrast = self.correct_contrast_brightness(crop_img, 2, 0)
    #cv2.imwrite("C:/Users/15ly1/PycharmProjects/TissueScanning/cropped_contrast.jpg", crop_img_contrast)

    # convert image to greyscale
    grey_crop = cv2.cvtColor(self.crop_img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('cropped image',grey_crop)
    cv2.imwrite('C:/Users/15ly1/PycharmProjects/TissueScanning/cropped_image_grey.jpg', grey_crop)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # threshold the image
    _, slide_threshold = cv2.threshold(grey_crop, 130, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # cv2.drawContours(crop_img, contours, -1, (0, 0, 255), 2)
    cv2.imwrite('C:/Users/15ly1/PycharmProjects/TissueScanning/slide_threshold.jpg', slide_threshold)

    # upscale image for better mm to pixel accuracy
    slide_threshold = cv2.resize(slide_threshold, (0, 0), fx=self.scaling_factor, fy=self.scaling_factor)

    # upscale the corners points by the same scaling factor
    self.corners = self.corners * self.scaling_factor

    # convert corner points to fit in the cropped image
    self.corners = self.calc_cropped_corners(self.corners)

    # calculate the size of the slide in pixels (height and width)
    slide_pixel_height, slide_pixel_width = self.calc_slide_size_pixels(self.corners)

    # calculate the number of pixels per micrometer (average the pixels per micrometer height and width)
    pixels_per_micrometer_width = slide_pixel_width / self.slide_microm_width
    pixels_per_micrometer_height = slide_pixel_height / self.slide_microm_height
    self.pixels_per_micrometer = (pixels_per_micrometer_height + pixels_per_micrometer_width) / 2
    print("Pixels per micrometer:", self.pixels_per_micrometer)
    print("Micrometer per pixel: ",
          ((self.slide_microm_height / slide_pixel_height) + (self.slide_microm_width / slide_pixel_width)) / 2)

    # find the contour of the tissue on the slide
    self.tissue_contour = self.find_tissue_contour(slide_pixel_width, slide_pixel_height, slide_threshold,
                                              self.pixels_per_micrometer)
    tissue_contour_image = cv2.resize(self.crop_img.copy(), (0, 0), fx=self.scaling_factor, fy=self.scaling_factor)
    cv2.drawContours(tissue_contour_image, self.tissue_contour, -1, (0, 255, 255), 5)
    cv2.imshow("tissue_contour", tissue_contour_image)
    print("tissue contour:", self.tissue_contour)
    cv2.imwrite("C:/Users/15ly1/PycharmProjects/TissueScanning/tissue_contour.jpg", tissue_contour_image)

    # generate a mask of the tissue contour
    self.tissue_mask = self.create_contour_mask(slide_threshold, self.tissue_contour)

    # cv2.imshow('tissue_mask', tissue_mask)
    cv2.imwrite('C:/Users/15ly1/PycharmProjects/TissueScanning/tissue_mask.jpg', self.tissue_mask)


  def generate_scanning_pattern(self):
    tissue_width, tissue_height, top_left_pt = self.calc_tissue_size(self.tissue_contour)
    print("tissue mask", self.tissue_mask)

    # calculate the linear scanning coordinates in pixels
    print("self pixels per micrometer:", self.pixels_per_micrometer)
    scan_pixel_img_coord, grid_pixel_img_coord = self.imaging(self.tissue_mask, tissue_height, tissue_width,
                                                              self.pixels_per_micrometer, self.step_size_microm, top_left_pt[0],
                                                              top_left_pt[1])

    # plot the grid coordinates in red and the scan_pixel coordinates in green for a visualization
    grid_vs_contour_lines = cv2.resize(self.crop_img.copy(), (0, 0), fx=self.scaling_factor, fy=self.scaling_factor)
    for grid_point in grid_pixel_img_coord:
      cv2.circle(grid_vs_contour_lines, tuple(grid_point), 1, (0, 0, 255))
    for contour_point in scan_pixel_img_coord:
      cv2.circle(grid_vs_contour_lines, tuple(contour_point), 1, (0, 255, 0))
    cv2.imwrite("C:/Users/15ly1/PycharmProjects/TissueScanning/resulting_points.jpg", grid_vs_contour_lines)
    # convert from image coordinate system to slide coordinate system
    scan_pixel_slide_coord = self.convert_to_slide_coordinate(scan_pixel_img_coord, self.corners)
    # convert from pixels to micrometers (divide the pixel coordinates matrix by pixels_per_micrometer conversion to find micrometers)
    scan_microm_slide_coord = np.divide(scan_pixel_slide_coord, self.pixels_per_micrometer)

    # calculate the size of scan pixels and grid pixels to calculate percentage change/difference
    contour_pixel_lines_rows = scan_pixel_img_coord.shape[0]
    grid_pixel_lines_rows = grid_pixel_img_coord.shape[0]

    # calculate the difference in size between grid scanning and contour scanning to show improvement
    percent_decrease = ((grid_pixel_lines_rows - contour_pixel_lines_rows) / grid_pixel_lines_rows) * 100


  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def isValidInputOutputData(self, inputVolumeNode, outputVolumeNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputVolumeNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    if inputVolumeNode.GetID()==outputVolumeNode.GetID():
      logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      return False
    return True

class Tissue_Scanning_ModuleTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_Tissue_Scanning_Module1()

  def test_Tissue_Scanning_Module1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import SampleData
    SampleData.downloadFromURL(
      nodeNames='FA',
      fileNames='FA.nrrd',
      uris='http://slicer.kitware.com/midas3/download?items=5767',
      checksums='SHA256:12d17fba4f2e1f1a843f0757366f28c3f3e1a8bb38836f0de2a32bb1cd476560')
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = Tissue_Scanning_ModuleLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
