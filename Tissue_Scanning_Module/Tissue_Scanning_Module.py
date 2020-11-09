import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np

scriptPath = os.path.dirname(os.path.abspath(__file__))
try:
  # the module is in the python path
  import cv2
except ImportError:
  # for the build directory, load from the file
  import imp, platform
  if platform.system() == 'Windows':
    cv2File = 'cv2.pyd'
    cv2Path = '../../../../OpenCV-build/lib/Release/' + cv2File
  else:
    cv2File = 'cv2.so'
    cv2Path = '../../../../OpenCV-build/lib/' + cv2File
  cv2Path = os.path.abspath(os.path.join(scriptPath, cv2Path))
  # in the build directory, this path should exist, but in the installed extension
  # it should be in the python path, so only use the short file name
  if not os.path.isfile(cv2Path):
    print('Full path not found: ',cv2Path)
    cv2Path = cv2File
  print('Loading cv2 from ',cv2Path)
  cv2 = imp.load_dynamic('cv2', cv2File)
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
    #uiWidget = slicer.util.loadUI(self.resourcePath('UI/Tissue_Scanning_Module.ui'))
    #self.layout.addWidget(uiWidget)
    #self.ui = slicer.util.childWidgetVariables(uiWidget)
#
# Parameters Area
        #
    CameraControlCollapsibleButton = ctk.ctkCollapsibleButton()
    CameraControlCollapsibleButton.text = "Camera Control "
    self.layout.addWidget(CameraControlCollapsibleButton)
    CameraControlFormLayout = qt.QFormLayout(CameraControlCollapsibleButton)
        #    
    
    #
        # IGT Link Connector
        #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLIGTLConnectorNode"]
    self.inputSelector.selectNodeUponCreation = False
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene(slicer.mrmlScene)
    self.inputSelector.setToolTip("Connect to OpenIGTLink to control printer from module.")
    CameraControlFormLayout.addRow("Connect to: ", self.inputSelector)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onCameraSelect)
    
    
    #self.ui.cameraInputSelector.setMRMLScene(slicer.mrmlScene)
    #self.ui.outputSelector.setMRMLScene(slicer.mrmlScene)

    # set max and min values of step size microm
    self.stepSizeNumSpinBox = qt.QDoubleSpinBox()
    self.stepSizeNumSpinBox.setMinimum(1)
    self.stepSizeNumSpinBox.setMaximum(10000)
    self.stepSizeNumSpinBox.setValue(500)
    CameraControlFormLayout.addRow("Step size (um) :", self.stepSizeNumSpinBox)
    # connections
    self.calibrateButton = qt.QPushButton("Calibrate Camera")
    self.calibrateButton.connect('clicked(bool)', self.onCalibrateButton)
    self.calibrateButton.enabled = True
    CameraControlFormLayout.addRow(self.calibrateButton)


    self.pictureButton = qt.QPushButton("Take Slide Picture")
    self.pictureButton.connect('clicked(bool)', self.onPictureButton)
    self.pictureButton.enabled = True
    CameraControlFormLayout.addRow(self.pictureButton)

    self.contourButton = qt.QPushButton("Determine Tissue Contour")
    self.contourButton.connect('clicked(bool)', self.onContourButton)
    self.contourButton.enabled = True
    CameraControlFormLayout.addRow(self.contourButton)
  
    self.scanningButton = qt.QPushButton("Generate Scanning Pattern")
    self.scanningButton.connect('clicked(bool)', self.onScanningButton)
    self.scanningButton.enabled = True
    CameraControlFormLayout.addRow(self.scanningButton)

    self.testButton = qt.QPushButton("Gnew button")
    self.testButton.connect('clicked(bool)', self.onTestButton)
    self.testButton.enabled = True
    CameraControlFormLayout.addRow(self.testButton)
    
    
    #self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onCameraSelect()

  def cleanup(self):
    pass

  def onCalibrateButton(self):
    pass

  def onCameraSelect(self):
    self.pictureButton.enabled = self.inputSelector.currentNode()
    self.calibrateButton.enabled = self.inputSelector.currentNode()
    self.logic.setCameraInputNode(self.inputSelector.currentNode())

  def onPictureButton(self):
    image = self.logic.takePicture()
    self.logic.setImage(image)
    print("Picture taken")
    self.contourButton.enabled = self.pictureButton.enabled
    #self.logic.find_slide_corners(image)
    #enableScreenshotsFlag = self.ui.enableScreenshotsFlagCheckBox.checked
    #imageThreshold = self.ui.imageThresholdSliderWidget.value
    #self.logic.run(self.ui.inputSelector.currentNode(), imageThreshold, enableScreenshotsFlag)

  def onContourButton(self):
    self.logic.determine_contour()
    self.scanningButton.enabled = self.contourButton.enabled

  def onScanningButton(self):
    self.logic.setStepSizeMicrom(self.stepSizeNumSpinBox.value)
    self.logic.generate_scanning_pattern()

  def onTestButton(self):
    self.logic.convert_array_to_fiducials()

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
    self.camera_matrix = np.array([ 3.6014714403014150e+03, 0., 6.4116609481666580e+02, 0.,
       3.6014714403014150e+03, 3.5745577227329079e+02, 0., 0., 1. ]).reshape(3, 3)
    self.distortion_coeff = np.array([ 2.5485041095855401e+00, -2.1526716048332378e+02,
       -1.6250023598792348e-02, 5.7989385007653493e-04,
       4.8914876856666788e+03 ])
    self.new_camera_mtx = []
    self.image = None
    self.crop_img = None
    self.tissue_contour = None
    self.tissue_mask = None
    # Set the slide width to 75mm (75000 micrometer)
    self.slide_microm_width = 75000
    self.slide_mm_width = 75
    # set the slide height to 25mm (25000 micrometer)
    self.slide_microm_height = 25000
    self.slide_mm_height = 25
    # set the step size to 50 micrometers
    self.step_size_microm = 10000 # will change with user input in spinbox
    # set robot resolution to 500 micrometers (0.5mm)
    self.robot_resolution = 50
    self.scaling_factor = 0
    self.pixels_per_micrometer = 0
    self.corners = []
    self.vtk_aruco_matrix = []
    self.numpy_aruco_matrix = []
    self.aruco_center_mm = []
    self.aruco_center_pixels = []
    self.aruco_corner_x_dist = 20
    self.aruco_corner_y_dist = 18
    self.genFidIndex = 0
    self.testFidIndex = 0
    self.scan_microm_slide_coord = []
    self.vtk_linear_transform_mtx = []
    self.corners_img = []



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

  def takePicture(self):
    image = slicer.util.array("Image_Image")
    image = image[0]
    image = self.undistort_camera(image)
    return image

  def takePicture_aruco(self):
    # get the name of the selected camera node
    #name = self.cameraInputNode
    # take screenshot of image and read it as an array & save the Aruco marker position
    image = slicer.util.array("Image_Image")
    aruco_transform = slicer.util.getNode("Marker5ToTracker")
    image = image[0]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    vtk_aruco_matrix = vtk.vtkMatrix4x4()
    aruco_transform.GetMatrixTransformToParent(vtk_aruco_matrix)
    self.vtk_aruco_matrix = vtk_aruco_matrix
    self.numpy_aruco_matrix = np.array(
      [vtk_aruco_matrix.GetElement(0, 0), vtk_aruco_matrix.GetElement(0, 1), vtk_aruco_matrix.GetElement(0, 2), vtk_aruco_matrix.GetElement(0, 3),
       vtk_aruco_matrix.GetElement(1, 0), vtk_aruco_matrix.GetElement(1, 1), vtk_aruco_matrix.GetElement(1, 2), vtk_aruco_matrix.GetElement(1, 3),
       vtk_aruco_matrix.GetElement(2, 0), vtk_aruco_matrix.GetElement(2, 1), vtk_aruco_matrix.GetElement(2, 2), vtk_aruco_matrix.GetElement(2, 3),
       vtk_aruco_matrix.GetElement(2, 0), vtk_aruco_matrix.GetElement(2, 1), vtk_aruco_matrix.GetElement(2, 2), vtk_aruco_matrix.GetElement(2, 3)]).reshape(4, 4)
    self.aruco_center_mm = np.array([vtk_aruco_matrix.GetElement(0,3), vtk_aruco_matrix.GetElement(1, 3), vtk_aruco_matrix.GetElement(2, 3)])
    print("Aruco mm", self.aruco_center_mm)
    normalize_aruco = np.array([self.aruco_center_mm[0]/self.aruco_center_mm[2], self.aruco_center_mm[1]/self.aruco_center_mm[2], 1]).reshape(3, 1)
    camera_matrix = self.camera_matrix
    self.aruco_center_pixels = np.dot(camera_matrix, normalize_aruco)
    print("aruco pixels",self.aruco_center_pixels)
    image = self.undistort_camera(image)
    cv2.circle(image, (self.aruco_center_pixels[0],self.aruco_center_pixels[1]), 5, (255, 0, 0), -1)
    cv2.imwrite("C:/Users/lconnolly/Desktop/use_this_tissue_scanning/center point.jpg", image)
    return image

  def undistort_camera(self, image):
    h, w = image.shape[:2]

    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coeff, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeff, None, new_camera_mtx)

    # crop the image

    #x, y, w, h = roi

    #undistorted_image = undistorted_image[y:y + h, x:x + w]
    return undistorted_image


  def midpoint(self, ptA, ptB):
    return ((ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2)

  def calc_dist_between_pts(self, ptA, ptB):
    x_diff = (ptB[0] - ptA[0]) ** 2
    y_diff = (ptB[1] - ptA[1]) ** 2
    return (math.sqrt(x_diff + y_diff))

  def convert_to_slide_coordinate(self, pixel_image_coordinates, slide_corners):
    x_image_vector = [1, 0]
    y_image_vector = [0, 1]
    translation_vector = [self.corners_img[0][0] - self.corners[0][0], self.corners_img[0][1] - self.corners[0][1], 0] 
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
    z_column = np.zeros((rows, 1))
    fourth_column = np.array([0,0,0,1]).reshape(4,1)
    print(fourth_column)
    pixel_coordinates_matrix = np.column_stack((np.column_stack((pixel_coordinates_matrix, z_column)),z_column))
    print("pixel_coordinates_matrix: \n", pixel_coordinates_matrix)
    transformation_matrix = np.array([[math.cos(theta), -1 * math.sin(theta), -1 * slide_corners[0][0]],
                                      [math.sin(theta), math.cos(theta), -1 * slide_corners[0][1]], [0, 0, 1]])
    transformation_matrix = np.row_stack((transformation_matrix, translation_vector))
    print("transformation Matrix after translation_vector:\n", transformation_matrix)
    transformation_matrix = np.column_stack((transformation_matrix, fourth_column))
    pixel_slide_coordinates_3D = transformation_matrix.dot(pixel_coordinates_matrix.T).T
    print("pixel slide coordinates:", pixel_slide_coordinates_3D)
    # remove z column before returning pixel_slide_coordinates to make coordinates 2D
    pixel_slide_coordinates_2D = np.delete(pixel_slide_coordinates_3D, 2, axis=1) # delete z axis
    pixel_slide_coordinates_2D = np.delete(pixel_slide_coordinates_2D, 2, axis=1) # delete extra 4th column
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
    y_num_step = round(input_tissue_height / pixels_per_step)
    x_num_step = round(input_tissue_width / pixels_per_step)
    print("number of y steps:", y_num_step)
    print("number of x steps:", x_num_step)
    y_step = 0

    # create a grid mask image for the lines to be printed on
    line_mask = np.zeros(input_contour_mask.shape, np.uint8)
    # calculate and draw the horizontal lines onto the mask image
    while y_step <= y_num_step:
      x_step = 0
      # calculate the points for the horizontal line moving vertically down the image
      y_axis = int(y_top_left + (y_step*pixels_per_step))
      while x_step <= x_num_step:
        x_axis = int(x_top_left + (x_step*pixels_per_step))
        point = tuple([x_axis, y_axis])
        cv2.circle(line_mask, point, 0, (255,255,255), -1)
        x_step += 1
      #horiz_left_point = (int(x_top_left), int(y_top_left + (step * pixels_per_step)))
      #horiz_right_point = (int(x_top_left + input_tissue_width), int(y_top_left + (step * pixels_per_step)))

      # print the points onto the image & plot the line
      # cv2.circle(line_mask, horiz_left_point, 5, (0,0,0), -1)
      # cv2.circle(line_mask, horiz_right_point, 5, (0,0,0), -1)
      #cv2.line(line_mask, horiz_left_point, horiz_right_point, (255, 255, 255), 1, cv2.LINE_AA)
      y_step += 1
    # cv2.imshow('mask', line_mask)
    cv2.imwrite('C:/Users/lconnolly/Desktop/use_this_tissue_scanning/line_mask.jpg', line_mask)
    # use numpy.logical_and to determine the pixels where the lines intersect the contour mask
    imaging_rows = cv2.bitwise_and(line_mask, input_contour_mask)
    imaging_rows = np.uint8(imaging_rows)
    #imaging_rows = cv2.cvtColor(imaging_rows, cv2.COLOR_BGR2GRAY)
    print(imaging_rows.shape)
    # cv2.imshow('contour lines', imaging_rows)
    cv2.imwrite('C:/Users/lconnolly/Desktop/use_this_tissue_scanning/contour_lines.jpg', imaging_rows)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # find the pixel locations of the contour scanning pattern
    contour_pixel_lines = cv2.findNonZero(imaging_rows)
    #print("contour_pixel_lines:", contour_pixel_lines)
    print("contour pixel lines (first 6):", contour_pixel_lines[0:6])


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
    mask = np.zeros(img.shape, dtype=np.uint8)
    # draw the contour points onto the mask image & fill it in
    cv2.drawContours(mask, [cnts], 0, (255, 255, 255), -1)
    # find the points within the contour
    # pixelpoints = cv2.findNonZero(mask)
    # cv2.imshow('mask', mask)
    return mask

  def calculate_aruco_marker_rotation_mtx(self):
    aruco_origin_x_direction_vector = np.array([1,0,0,0]).reshape(4,1)
    aruco_origin_y_direction_vector = np.array([0,1,0,0]).reshape(4,1)
    aruco_origin_z_direction_vector = np.array([0,0,1,0]).reshape(4,1)
    transformation_matrix = self.vtk_aruco_matrix
    camera_aruco_x_vector = transformation_matrix.MultiplyPoint(aruco_origin_x_direction_vector)
    camera_aruco_y_vector = transformation_matrix.MultiplyPoint(aruco_origin_y_direction_vector)
    camera_aruco_z_vector = transformation_matrix.MultiplyPoint(aruco_origin_z_direction_vector)
    norm_camera_aruco_x = np.array([camera_aruco_x_vector[0], camera_aruco_x_vector[1], 0]).reshape(3, 1)
    norm_camera_aruco_y = np.array([camera_aruco_y_vector[0], camera_aruco_y_vector[1], 0]).reshape(3, 1)
    norm_camera_aruco_z = np.array([camera_aruco_z_vector[0], camera_aruco_z_vector[1], 0]).reshape(3, 1)
    rotation_matrix = np.array(np.hstack([norm_camera_aruco_x, norm_camera_aruco_y, norm_camera_aruco_z]))
    return rotation_matrix

  def create_contour_mask(self, img, cnts):
    # create the mask image with all zeros
    mask = np.zeros(img.shape, dtype=np.uint8)
    # draw the contour points onto the mask image & fill it in
    cv2.drawContours(mask, [cnts], 0, (255, 255, 255), -1)
    # find the points within the contour
    #pixelpoints = cv2.findNonZero(mask)
    #cv2.imshow('mask', mask)
    return mask

  def find_corners(self, input_image, input_contours):
    height, width, _ = input_image.shape
    bw_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    image_area = height*width
    corners_image = input_image.copy()
    # search for the slide contour through the list of contours
    for contour in input_contours:
        size = cv2.contourArea(contour)
        # if the contour is of reasonable size (the slide contour) then create a mask
        if size > image_area/10:
            mask = self.create_contour_mask(bw_input_image, contour)
    #cv2.imshow('contours mask', mask)
    # find all non-zero points in the form [x,y] (column, row)
    cv2.imwrite("C:/Users/lconnolly/Desktop/use_this_tissue_scanning/tissue_mask.jpg", mask)
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
        cv2.circle(corners_image, (point[0], point[1]), 3, (0, 255, 255), -1)

    #cv2.imshow('mask w/ corners', input_image)
    cv2.imwrite('C:/Users/lconnolly/Desktop/use_this_tissue_scanning/calculated_corners.jpg', corners_image)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    return np.asarray(points)

# NOT USED ANYMORE
  def find_slide_corners(self, input_image):
    top_right_relative_aruco = [self.aruco_corner_x_dist, -1*self.aruco_corner_y_dist, 0, 1]
    top_left_relative_aruco = [self.aruco_corner_x_dist, -1*(self.aruco_corner_y_dist + self.slide_mm_width), 0, 1]
    bottom_right_relative_aruco = [self.aruco_corner_x_dist + self.slide_mm_height, -1*self.aruco_corner_y_dist, 0, 1]
    bottom_left_relative_aruco = [self.aruco_corner_x_dist + self.slide_mm_height, -1*(self.aruco_corner_y_dist + self.slide_mm_width), 0, 1]
    top_left_mm = np.array(self.vtk_aruco_matrix.MultiplyPoint(top_left_relative_aruco))
    top_right_mm = np.array(self.vtk_aruco_matrix.MultiplyPoint(top_right_relative_aruco))
    bottom_left_mm = np.array(self.vtk_aruco_matrix.MultiplyPoint(bottom_left_relative_aruco))
    bottom_right_mm = np.array(self.vtk_aruco_matrix.MultiplyPoint(bottom_right_relative_aruco))
    # normalize mm points to convert to pixels
    norm_top_left_mm = np.array([top_left_mm[0]/top_left_mm[2], top_left_mm[1]/top_left_mm[2], 1]).reshape(3, 1)
    #print(norm_top_left_mm)
    norm_top_right_mm = np.array([top_right_mm[0]/top_right_mm[2], top_right_mm[1]/top_right_mm[2], 1]).reshape(3, 1)
    norm_bottom_left_mm = np.array([bottom_left_mm[0]/bottom_left_mm[2], bottom_left_mm[1]/bottom_left_mm[2], 1]).reshape(3, 1)
    norm_bottom_right_mm = np.array([bottom_right_mm[0]/bottom_right_mm[2], bottom_right_mm[1]/bottom_right_mm[2], 1]).reshape(3, 1)
    camera_matrix = self.camera_matrix.reshape(3, 3)
    top_left_pixels = np.dot(camera_matrix, norm_top_left_mm)
    top_right_pixels = np.dot(camera_matrix, norm_top_right_mm)
    bottom_left_pixels = np.dot(camera_matrix, norm_bottom_left_mm)
    bottom_right_pixels = np.dot(camera_matrix, norm_bottom_right_mm)
    print("top left pixels:", top_left_pixels)
    cv2.circle(input_image, (top_left_pixels[0], top_left_pixels[1]), 5, (255, 0, 0), -1)
    cv2.circle(input_image, (top_right_pixels[0], top_right_pixels[1]), 5, (0, 255, 0), -1)
    cv2.circle(input_image, (bottom_left_pixels[0], bottom_left_pixels[1]), 5, (0, 0, 255), -1)
    cv2.circle(input_image, (bottom_right_pixels[0], bottom_right_pixels[1]), 5, (0, 255, 255), -1)
    cv2.imwrite("C:/Users/lconnolly/Desktop/use_this_tissue_scanning/center point.jpg", input_image)
    return np.array([top_left_pixels.reshape(1,3), bottom_right_pixels.reshape(1,3), bottom_left_pixels.reshape(1,3), top_right_pixels.reshape(1,3)])

  def find_tissue_contour(self, slide_width, slide_height, slide_threshold, pixels_per_micrometer):
    # calculate size of photo
    height, width = slide_threshold.shape
    print("Height of slide:", height)
    print("Width of slide:", width)
    # calculate the area of the slide
    slide_area = slide_width * slide_height
    # threshold the slide image
    _, tissue_contours, hierarchy = cv2.findContours(slide_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("tissue_contours", tissue_contours[0][0])
    for contour in tissue_contours:
      image = self.crop_img.copy()
      print "entering for loop"
      # check if contour is touching the border of the image (tissue will not be placed on edge of slide)
      bounding_x, bounding_y, bounding_width, bounding_height = cv2.boundingRect(contour)
      if bounding_x >= 0 and bounding_y >= 0 and bounding_x + bounding_width <= width - 5 and bounding_y + bounding_height <= height - 5:
        print("entering bounding box if statement")
        # check that the area of the contour is greater than 750000 micrometers and less than 375000000 micrometers
        if (slide_area * 0.004) / pixels_per_micrometer < cv2.contourArea(contour) / pixels_per_micrometer:
          #cv2.rectangle(image, (bounding_x, bounding_y), (bounding_x + bounding_width, bounding_y + bounding_height), (2,255,0),2)
          #cv2.drawContours(image, contour, -1, (0, 255, 255), 2)
          #cv2.imshow("rectangle", image)
          #cv2.waitKey()
          #cv2.destroyAllWindows()
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
    print("corners[0]:", corners[0])
    print("corners[1]:", corners[1])
    print("corners[2]:", corners[2])
    print("corners[3]:", corners[3])
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

  def fiducialMarker(self, xcoordinate, ycoordinate, zcoordinate):
    self.fiducialNode = slicer.vtkMRMLMarkupsFiducialNode()
    slicer.mrmlScene.AddNode(self.fiducialNode)
    self.fiducialNode.SetName("")
    self.fiducialNode.SetNthFiducialLabel(0, "")
    self.fiducialNode.AddFiducial(xcoordinate, ycoordinate, zcoordinate)

  def addToCurrentFiducialNode(self, xcoordinate, ycoordinate, zcoordinate):
    self.fiducialNode.AddFiducial(xcoordinate, ycoordinate, zcoordinate)
    #self.fiducialNode.SetNthFiducialLabel(self.genFidIndex, "")
    self.genFidIndex= self.genFidIndex + 1

  def convert_array_to_fiducials(self):
    linear_transform_node = slicer.util.getNode("LinearTransform_3")
    vtk_linear_transform_mtx = vtk.vtkMatrix4x4()
    linear_transform_node.GetMatrixTransformToParent(vtk_linear_transform_mtx)
    print(vtk_linear_transform_mtx)
    self.vtk_linear_transform_mtx = vtk_linear_transform_mtx
    pts = np.load(r"C:\Users\lconnolly\Desktop\use_this_tissue_scanning\contour points.npy")
    #pts = np.divide(pts, 1000)
    #pts = self.scan_microm_slide_coord
    transformed_pts = []
    #print("outside for loop")
    #print(pts)
    for i in range(0,len(pts)-1):
      
      pt_4x1 = np.array([pts[i][0], pts[i][1], 0, 1]).reshape(4,1)
      new_pt = np.array(self.vtk_linear_transform_mtx.MultiplyPoint(pt_4x1))
      #print(new_pt[0:2])
      #np.append(transformed_pts, new_pt)
      #transformed_pts[i] = new_pt
      #print(transformed_pts[i])
      transformed_pts.append(new_pt[0:2])
    transformed_np_pts = np.asarray(transformed_pts)
    #print(transformed_pts)
    print("transformed Points:", transformed_np_pts)
    np.save(r"C:/Users/lconnolly/Desktop/use_this_tissue_scanning/transformed pts.npy", transformed_np_pts)

    

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

    # covert image to greyscale
    
    #self.image = cv2.imread(r"C:\Users\lconnolly\Desktop\use_this_tissue_scanning\image_black_and_white.jpg")
    img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("C:/Users/lconnolly/Desktop/use_this_tissue_scanning/image_black_and_white.jpg", img)
    # cv2.imshow('image', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # calculate the threshold
    _, threshold = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("C:/Users/lconnolly/Desktop/use_this_tissue_scanning/image_threshold.jpg", threshold)

    # cv2.imshow('threshold',threshold)
    # cv2.imshow('inv_threshold',inv_threshold)

    # Find contour of the tissue slide
    _, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours onto display image
    contours_copy = self.image.copy()

    # calculate the corners of the slide
    self.corners = self.find_corners(contours_copy, contours)
    #self.corners = self.find_slide_corners(contours_copy)
    print(self.corners)
    self.corners = np.vstack(self.corners).squeeze()
    print(self.corners)
    #self.corners = self.find_corners(contours_copy, contours)
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
    print("np min column 1", math.floor(np.min(self.corners[:, 1])))
    print("np max column 1", math.ceil(np.max(self.corners[:,1])))
    self.crop_img = self.image[int(math.floor(np.min(self.corners[:, 1]))):int(math.ceil(np.amax(self.corners[:, 1]))),
                    int(math.floor(np.amin(self.corners[:, 0]))):int(math.ceil(np.amax(self.corners[:, 0])))]
    cv2.imwrite("C:/Users/lconnolly/Desktop/use_this_tissue_scanning/cropped_image.jpg", self.crop_img)

    # increase contrast of image to brighten tissue sample
    #crop_img_contrast = self.correct_contrast_brightness(crop_img, 2, 0)
    #cv2.imwrite("C:/Users/15ly1/PycharmProjects/TissueScanning/cropped_contrast.jpg", crop_img_contrast)

    # convert image to greyscale
    grey_crop = cv2.cvtColor(self.crop_img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('cropped image',grey_crop)
    cv2.imwrite('C:/Users/lconnolly/Desktop/use_this_tissue_scanning/cropped_image_grey.jpg', grey_crop)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # threshold the image
    _, slide_threshold = cv2.threshold(grey_crop, 130, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # cv2.drawContours(crop_img, contours, -1, (0, 0, 255), 2)
    cv2.imwrite('C:/Users/lconnolly/Desktop/use_this_tissue_scanning/slide_threshold.jpg', slide_threshold)

    # upscale image for better mm to pixel accuracy
    slide_threshold = cv2.resize(slide_threshold, (0, 0), fx=self.scaling_factor, fy=self.scaling_factor)

    # upscale the corners points by the same scaling factor
    self.corners_img = self.corners * self.scaling_factor

    # convert corner points to fit in the cropped image
    self.corners = self.calc_cropped_corners(self.corners_img)
    print(self.corners)

  ## Fiducial node generation
    #self.dataCollection = self.createPolyDataPoint(self.xcoordinate, self.ycoordinate, self.zcoordinate)
    self.fiducialMarker(self.corners[0][0], self.corners[0][1], 0)
    self.genFidIndex= self.genFidIndex + 1
    self.addToCurrentFiducialNode(self.corners[1][0], self.corners[1][1], 0)
    self.addToCurrentFiducialNode(self.corners[2][0], self.corners[2][1], 0)
    self.addToCurrentFiducialNode(self.corners[3][0], self.corners[3][1], 0)
    


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
    self.tissue_contour = self.find_tissue_contour(slide_pixel_width, slide_pixel_height, slide_threshold, self.pixels_per_micrometer)
    print("tissue_contour_found")
    tissue_contour_image = cv2.resize(self.crop_img.copy(), (0, 0), fx=self.scaling_factor, fy=self.scaling_factor)
    cv2.drawContours(tissue_contour_image, self.tissue_contour, -1, (0, 255, 255), 5)
    cv2.imshow("tissue_contour", tissue_contour_image)
    print("tissue contour:", self.tissue_contour)
    cv2.imwrite("C:/Users/lconnolly/Desktop/use_this_tissue_scanning/tissue_contour.jpg", tissue_contour_image)

    # generate a mask of the tissue contour
    self.tissue_mask = self.create_contour_mask(slide_threshold, self.tissue_contour)

    # cv2.imshow('tissue_mask', tissue_mask)
    cv2.imwrite('C:/Users/lconnolly/Desktop/use_this_tissue_scanning/tissue_mask.jpg', self.tissue_mask)


  def generate_scanning_pattern(self):
    tissue_width, tissue_height, top_left_pt = self.calc_tissue_size(self.tissue_contour)

    # calculate the linear scanning coordinates in pixels
    scan_pixel_img_coord, grid_pixel_img_coord = self.imaging(self.tissue_mask, tissue_height, tissue_width,
                                                              self.pixels_per_micrometer, self.step_size_microm, top_left_pt[0],
                                                              top_left_pt[1])

    # plot the grid coordinates in red and the scan_pixel coordinates in green for a visualization
    grid_vs_contour_lines = cv2.resize(self.crop_img.copy(), (0, 0), fx=self.scaling_factor, fy=self.scaling_factor)
    for grid_point in grid_pixel_img_coord:
      cv2.circle(grid_vs_contour_lines, tuple(grid_point), 1, (0, 0, 255))
    for contour_point in scan_pixel_img_coord:
      cv2.circle(grid_vs_contour_lines, tuple(contour_point), 1, (0, 255, 0))
    cv2.imwrite("C:/Users/lconnolly/Desktop/use_this_tissue_scanning/resulting_points.jpg", grid_vs_contour_lines)
    cv2.imshow("grid vs contour lines", grid_vs_contour_lines)
    # convert from image coordinate system to slide coordinate system
    #scan_pixel_slide_coord = self.convert_to_slide_coordinate(scan_pixel_img_coord, self.corners)
    #print("scan_pixel_img_coord:", scan_pixel_img_coord)
    #print("scan_pixel_slide_coord:", scan_pixel_slide_coord)
    # convert from pixels to micrometers (divide the pixel coordinates matrix by pixels_per_micrometer conversion to find micrometers)
    #self.scan_microm_slide_coord = np.divide(scan_pixel_slide_coord, self.pixels_per_micrometer)
    np.save(r"C:/Users/lconnolly/Desktop/use_this_tissue_scanning/contour points.npy", scan_pixel_img_coord)

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
