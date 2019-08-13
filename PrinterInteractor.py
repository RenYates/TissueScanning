import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import time
import functools
import argparse
import math
import numpy as np


#
# PrinterInteractor
#

class PrinterInteractor(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "PrinterInteractor"  # TODO make this more human readable by adding spaces
        self.parent.categories = ["SlicerSpectroscopy"]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Laura Connolly PerkLab (Queen's University), Mark Asselin PerkLab (Queen's University)"]  # replace with "Firstname Lastname (Organization)"
        self.parent.helpText = """
This is an module developed to interface Slicer Software with the Monoprice Mini V2 3D Printer for spectroscopy 
"""
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""  # replace with organization, grant and thanks.


#
# PrinterInteractorWidget
#

class PrinterInteractorWidget(ScriptedLoadableModuleWidget):
    """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def setup(self):


        ScriptedLoadableModuleWidget.setup(self)

        # Instantiate and connect widgets ...
        self.logic = PrinterInteractorLogic()
        #
        # Parameters Area
        #
        PrinterControlCollapsibleButton = ctk.ctkCollapsibleButton()
        PrinterControlCollapsibleButton.text = "Printer Control "
        self.layout.addWidget(PrinterControlCollapsibleButton)
        PrinterControlFormLayout = qt.QFormLayout(PrinterControlCollapsibleButton)
        #
        # ROI Systematic Search Area
        #
        ROICollapsibleButton = ctk.ctkCollapsibleButton()
        ROICollapsibleButton.text = " Optimized Scanning Tools"
        ROICollapsibleButton.collapsed = True
        self.layout.addWidget(ROICollapsibleButton)
        ROIFormLayout = qt.QFormLayout(ROICollapsibleButton)
        #
        # Contour Tracing Tool Area
        #
        ContourTracingCollapsibleButton = ctk.ctkCollapsibleButton()
        ContourTracingCollapsibleButton.text = " Contour Tracing Tools"
        ContourTracingCollapsibleButton.collapsed = True
        self.layout.addWidget(ContourTracingCollapsibleButton)
        ContourTracingFormLayout = qt.QFormLayout(ContourTracingCollapsibleButton)
        #
        # Image Registration Tool Area
        #
        ImageRegistrationCollapsibleButton = ctk.ctkCollapsibleButton()
        ImageRegistrationCollapsibleButton.text = " Image Registration Tools"
        ImageRegistrationCollapsibleButton.collapsed = True
        self.layout.addWidget(ImageRegistrationCollapsibleButton)
        ImageRegistrationFormLayout = qt.QFormLayout(ImageRegistrationCollapsibleButton)
        #
        # Wavelength Selector
        #
        self.probeSelector = qt.QComboBox()
        self.probeSelector.insertItem(1, "RED: 660 nm")
        #self.probeSelector.insertItem(2, "UV: 395 nm ")
        PrinterControlFormLayout.addRow("Laser Wavelength :", self.probeSelector)
        #
        # Learn Spectra Button
        #
        self.learnSpectraButton = qt.QPushButton("Learn Spectra")
        self.learnSpectraButton.toolTip = "Move over spectra of interest to collect reference."
        self.learnSpectraButton.enabled = True
        PrinterControlFormLayout.addRow(self.learnSpectraButton)
        self.learnSpectraButton.connect('clicked(bool)', self.onLearnSpectraButton)
        #
        # Home Button
        #
        self.homeButton = qt.QPushButton("Home")
        self.homeButton.toolTip = "Return to reference axis"
        self.homeButton.enabled = True
        PrinterControlFormLayout.addRow(self.homeButton)
        self.homeButton.connect('clicked(bool)', self.onHomeButton)
        #
        # Keyboard ShortCut Button
        #
        self.shortcutButton = qt.QPushButton("Activate Keyboard Shortcuts")
        self.shortcutButton.toolTip = "Activate arrow key movement shortcuts."
        self.shortcutButton.enabled = True
        PrinterControlFormLayout.addRow(self.shortcutButton)
        self.shortcutButton.connect('clicked(bool)', self.onActivateKeyboardShortcuts)
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
        PrinterControlFormLayout.addRow("Connect to: ", self.inputSelector)
        #
        # Output Array Selector
        #
        self.outputArraySelector = slicer.qMRMLNodeComboBox()
        self.outputArraySelector.nodeTypes = (("vtkMRMLDoubleArrayNode"), "")
        self.outputArraySelector.addEnabled = True
        self.outputArraySelector.removeEnabled = True
        self.outputArraySelector.noneEnabled = False
        self.outputArraySelector.showHidden = False
        self.outputArraySelector.showChildNodeTypes = False
        self.outputArraySelector.setMRMLScene(slicer.mrmlScene)
        self.outputArraySelector.setToolTip("Pick the output array for spectrum analysis.")
        PrinterControlFormLayout.addRow("Output spectrum array: ", self.outputArraySelector)
        #
        # X Resolution
        #
        self.xResolution_spinbox = qt.QDoubleSpinBox()
        self.xResolution_spinbox.setMinimum(0)
        self.xResolution_spinbox.setMaximum(120)
        self.xResolution_spinbox.setValue(10)
        PrinterControlFormLayout.addRow("X resolution (mm / step) :", self.xResolution_spinbox)
        #
        # Y Resolution
        #
        self.yResolution_spinbox = qt.QDoubleSpinBox()
        self.yResolution_spinbox.setMinimum(0)
        self.yResolution_spinbox.setMaximum(120)
        self.yResolution_spinbox.setValue(10)
        PrinterControlFormLayout.addRow("Y resolution (mm/ step):", self.yResolution_spinbox)
        #
        # Z movement
        #
        self.zResolution_spinbox = qt.QDoubleSpinBox()
        self.zResolution_spinbox.setMinimum(-20)
        self.zResolution_spinbox.setMaximum(150)
        self.zResolution_spinbox.setValue(10)
        PrinterControlFormLayout.addRow("Z Position (distance from 0):", self.zResolution_spinbox)
        #
        # Time per Reading
        #
        self.timeDelay_spinbox = qt.QSpinBox()
        self.timeDelay_spinbox.setMinimum(0)
        self.timeDelay_spinbox.setMaximum(5000)
        self.timeDelay_spinbox.setValue(1000)
        PrinterControlFormLayout.addRow("Time for data delay (ms) :", self.timeDelay_spinbox)
        #
        # Fiducial Placement on/ off
        #
        self.fiducialMarkerCheckBox = qt.QCheckBox()
        self.fiducialMarkerCheckBox.checked = 0
        PrinterControlFormLayout.addRow("Fiducial Marking Off:", self.fiducialMarkerCheckBox)
        self.fiducialMarkerCheckBox.connect('stateChanged(int)', self.onFiducialMarkerChecked)
        #
        # Z Movement
        #
        self.verticalControlButton = qt.QPushButton("Vertical Control")
        self.verticalControlButton.toolTip = "Move over spectra of interest to collect reference."
        self.verticalControlButton.enabled = True
        PrinterControlFormLayout.addRow(self.verticalControlButton)
        self.verticalControlButton.connect('clicked(bool)', self.onZResolutionButton)
        #
        # Surface Scan Button
        #
        self.scanButton = qt.QPushButton("GO")
        self.scanButton.toolTip = "Begin systematic surface scan"
        self.scanButton.enabled = True
        PrinterControlFormLayout.addRow(self.scanButton)
        self.scanButton.connect('clicked(bool)', self.onScanButton)
        self.scanButton.setStyleSheet("background-color: green; font: bold")
        #
        # Surface Scan Button
        #
        self.patternButton = qt.QPushButton("Follow Pattern")
        self.patternButton.toolTip = "Begin systematic surface scan"
        self.patternButton.enabled = True
        PrinterControlFormLayout.addRow(self.patternButton)
        self.patternButton.connect('clicked(bool)', self.onPatternButton)
        self.patternButton.setStyleSheet("background-color: green; font: bold")
        #
        # Stop Button
        #
        self.stopButton = qt.QPushButton("STOP")
        self.stopButton.toolTip = "Requires restart (slicer and printer)."
        self.stopButton.enabled = True
        PrinterControlFormLayout.addRow(self.stopButton)
        self.stopButton.connect('clicked(bool)', self.onStopButton)
        self.stopButton.setStyleSheet("background-color: red; font: bold")
        #
        # Place boundary point
        #
        self.placeBoundaryButton = qt.QPushButton("Place Boundary Fiducial")
        self.placeBoundaryButton.toolTip = " Calculate and move to the center of mass of a ROI indicated by fiducials"
        self.placeBoundaryButton.enabled = True
        ROIFormLayout.addRow(self.placeBoundaryButton)
        self.placeBoundaryButton.connect('clicked(bool)', self.onPlaceBoundaries)
        #
        # ROI Systematic Search Button
        #
        self.ROIsearchButton = qt.QPushButton("ROI Rectilinear Scan")
        self.ROIsearchButton.toolTip = " "
        self.ROIsearchButton.enabled = True
        ROIFormLayout.addRow(self.ROIsearchButton)
        self.ROIsearchButton.connect('clicked(bool)', self.ROIsearch)
        #
        # ROI Raster Search
        #
        self.ROIrasterButton = qt.QPushButton("ROI Raster Scan")
        self.ROIrasterButton.toolTip = " "
        self.ROIrasterButton.enabled = True
        ROIFormLayout.addRow(self.ROIrasterButton)
        self.ROIrasterButton.connect('clicked(bool)', self.ROIrastersearch)
        #
        # Edge Tracing Button
        #
        self.ConvexHullTraceButton = qt.QPushButton("Trace Contour (after systematic scan)")
        self.ConvexHullTraceButton.toolTip = "Outline the image."
        self.ConvexHullTraceButton.enabled = True
        ContourTracingFormLayout.addRow(self.ConvexHullTraceButton)
        self.ConvexHullTraceButton.connect('clicked(bool)', self.onFindConvexHull)
        #
        # Edge Locating Button
        #
        self.edgeLocatorButton = qt.QPushButton("Find Edge")
        self.edgeLocatorButton.toolTip = "Move to the edge of the area of interest."
        self.edgeLocatorButton.enabled = True
        ContourTracingFormLayout.addRow(self.edgeLocatorButton)
        self.edgeLocatorButton.connect('clicked(bool)', self.onFindEdge)
        #
        # Quadrant Resolution
        #
        self.quadResolution_spinbox = qt.QDoubleSpinBox()
        self.quadResolution_spinbox.setMinimum(0)
        self.quadResolution_spinbox.setMaximum(10)
        self.quadResolution_spinbox.setValue(5)
        ContourTracingFormLayout.addRow("Quadrant Searching Resolution (mm/ step):", self.quadResolution_spinbox)
        #
        # Independent Contour Trace Button
        #
        self.independentEdgeTraceButton = qt.QPushButton("Trace Contour (after edge found, without systematic scan)")
        self.independentEdgeTraceButton.toolTip = "Independent contour tracing using a root finding algorithm."
        self.independentEdgeTraceButton.enabled = True
        ContourTracingFormLayout.addRow(self.independentEdgeTraceButton)
        self.independentEdgeTraceButton.connect('clicked(bool)', self.onIndependentContourTrace)
        #
        # Instruction area
        #
        self.textLine = qt.QTextEdit()
        self.textLine.setPlainText("Note: If transformation is not showing up correctly in slice views,"
                                   " check transform hierarchy in Data module and Volume Reslice Driver.")
        self.textLine.setReadOnly(1)
        ImageRegistrationFormLayout.addRow(self.textLine)
        #
        # Place Registration Point
        #
        self.placeRegistrationPtButton = qt.QPushButton("Place Optical Landmark")
        self.placeRegistrationPtButton.toolTip = "Place landmark fiducials visible in optical space."
        self.placeRegistrationPtButton.enabled = True
        ImageRegistrationFormLayout.addRow(self.placeRegistrationPtButton)
        self.placeRegistrationPtButton.connect('clicked(bool)', self.onPlaceFiducials)
        #
        # Landmark registration Button
        #
        self.landmarkRegButton = qt.QPushButton("Landmark Registration")
        self.landmarkRegButton.enabled = True
        ImageRegistrationFormLayout.addRow(self.landmarkRegButton)
        self.landmarkRegButton.connect('clicked(bool)', self.onLandmarkRegButton)

        self.layout.addStretch(1)

    def cleanup(self):
        pass

    def onSerialIGLTSelectorChanged(self):
        self.logic.setSerialIGTLNode(serialIGTLNode=self.inputSelector.currentNode())
        pass

    def ondoubleArrayNodeChanged(self):
        self.logic.setdoubleArrayNode(doubleArrayNode=self.inputSelector.currentNode())
        pass

    def onLearnSpectraButton(self):
        self.ondoubleArrayNodeChanged()
        self.onSerialIGLTSelectorChanged()
        self.logic.getSpectralData(self.outputArraySelector.currentNode())

    def onHomeButton(self, SerialIGTLNode):
        self.onSerialIGLTSelectorChanged()
        self.logic.home()

    def onActivateKeyboardShortcuts(self):
        self.logic.declareShortcut(serialIGTLNode=self.inputSelector.currentNode())
        print "Shortcuts activated."

    def onScanButton(self):
        
        # Printer Movement
        self.onSerialIGLTSelectorChanged()
        self.mvmtDelay = self.timeDelay_spinbox.value
        xResolution = self.xResolution_spinbox.value
        yResolution = self.yResolution_spinbox.value
        if xResolution < 2 or yResolution < 2:
            print "Error: Resolution too high. Try ROI systematic scanning."
            return
        else:
            self.logic.xLoop(self.mvmtDelay, xResolution, yResolution)
            self.logic.yLoop(self.mvmtDelay, yResolution, xResolution)
        
            stopsToVisitX = 120 / xResolution
            stopsToVisitY = 120 / yResolution
        
        # Spectrum Analysis
            self.tissueAnalysisTimer = qt.QTimer()
            self.iterationTimingValue = 0
        
            for self.iterationTimingValue in self.logic.frange(0, (stopsToVisitX * stopsToVisitY * self.mvmtDelay) + (10 * self.mvmtDelay), self.mvmtDelay):
                self.tissueAnalysisTimer.singleShot(self.iterationTimingValue, lambda: self.tissueDecision())
                self.iterationTimingValue = self.iterationTimingValue + self.mvmtDelay

    def onPatternButton(self, SerialIGTLNode):
        self.onSerialIGLTSelectorChanged()
        self.logic.contour_pattern()

    def tissueDecision(self):
        self.ondoubleArrayNodeChanged()
        if self.logic.spectrumComparison(self.outputArraySelector.currentNode()) == False:  # add a fiducial if the the tumor detecting function returns false
            self.logic.get_coordinates()

    def onFiducialMarkerChecked(self):
        # Turns off fiducial Marking when checked
        self.logic.fiducialMarkerChecked()

    def onZResolutionButton(self):
        self.ondoubleArrayNodeChanged()
        self.onSerialIGLTSelectorChanged()
        zValue = self.zResolution_spinbox.value
        self.logic.controlledZMovement(zValue)

    def onStopButton(self):
        self.onSerialIGLTSelectorChanged()
        self.logic.emergencyStop()
        # Note: the stop command uses G-code command M112 which requires slicer reboot and printer reboot after each usage.

    def onPlaceBoundaries(self):
        self.ondoubleArrayNodeChanged()
        self.onSerialIGLTSelectorChanged()
        self.logic.getBoundaryFiducialsCoordinate()

    def ROIsearch(self):
        # Printer Movement
        self.ondoubleArrayNodeChanged()
        self.onSerialIGLTSelectorChanged()
        self.mvmtDelay = self.timeDelay_spinbox.value
        xResolution = self.xResolution_spinbox.value
        yResolution = self.yResolution_spinbox.value
        if self.logic.ROIBoundarySearch() == False:
            return
        xMin, xMax, yMin, yMax = self.logic.ROIBoundarySearch()
        self.logic.yMovement(0, yMin)
        self.logic.XMovement(0, xMin)
        self.logic.ROIsearchXLoop(self.mvmtDelay, xResolution, yResolution, xMin,xMax, yMin, yMax)
        self.logic.ROIsearchYLoop(self.mvmtDelay, yResolution, xResolution, yMin, yMax, xMin,xMax)

        # Tissue Analysis
        self.tissueAnalysisTimer = qt.QTimer()
        self.iterationTimingValue = 0


        stopsToVisitX = (xMax - xMin) / xResolution
        stopsToVisitY = (yMax - yMin) / yResolution
        for self.iterationTimingValue in self.logic.frange(0, (stopsToVisitX * stopsToVisitY * self.mvmtDelay) + 16 * self.mvmtDelay,self.mvmtDelay):
            self.tissueAnalysisTimer.singleShot(self.iterationTimingValue, lambda: self.tissueDecision())
            self.iterationTimingValue = self.iterationTimingValue + self.mvmtDelay


    def ROIrastersearch(self):
        self.ondoubleArrayNodeChanged()
        self.onSerialIGLTSelectorChanged()
        xResolution = self.xResolution_spinbox.value
        yResolution = self.yResolution_spinbox.value
        delay = self.timeDelay_spinbox.value

        self.logic.zigzagPattern(xResolution, yResolution, delay)

        # Tissue Analysis
        self.tissueAnalysisTimer = qt.QTimer()
        self.iterationTimingValue = 0

        xMin, xMax, yMin, yMax = self.logic.ROIBoundarySearch()
        self.mvmtDelay = self.timeDelay_spinbox.value
        stopsToVisitX = (xMax - xMin) / xResolution
        stopsToVisitY = (yMax - yMin) / yResolution
        for self.iterationTimingValue in self.logic.frange(0, (stopsToVisitX * stopsToVisitY * self.mvmtDelay) + 16 * self.mvmtDelay,self.mvmtDelay):
            self.tissueAnalysisTimer.singleShot(self.iterationTimingValue, lambda: self.tissueDecision())

            self.iterationTimingValue = self.iterationTimingValue + self.mvmtDelay

    def onFindConvexHull(self):
        self.logic.convexHull()

    def onFindEdge(self):
        # Use to find the edge of the area of interest before independent edge tracing
        self.ondoubleArrayNodeChanged()
        self.onSerialIGLTSelectorChanged()
        self.logic.findAndMoveToEdge(self.outputArraySelector.currentNode())

    def onIndependentContourTrace(self):
        self.ondoubleArrayNodeChanged()
        self.onSerialIGLTSelectorChanged()
        quadrantResolution = self.quadResolution_spinbox.value
        self.logic.edgeTrace(self.outputArraySelector.currentNode(), quadrantResolution)

    def onPlaceFiducials(self):
        self.ondoubleArrayNodeChanged()
        self.onSerialIGLTSelectorChanged()
        self.logic.getLandmarkFiducialsCoordinate()

    def onLandmarkRegButton(self):
        self.logic.landmarkRegistration()


#
# PrinterInteractorLogic
#

class PrinterInteractorLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
    # arrays for convex Hull
    _yHullArray = []
    _xHullArray = []
    # arrays for edge tracing
    _saveycoordinate = []
    _savexcoordinate = []
    # arrays for quadrant check (independent edge tracing)
    _tumorCheck = []
    # ROI boundary arrays
    _ROIxbounds = []
    _ROIybounds = []

    def __init__(self):

        # General instantiations
        self.serialIGTLNode = None
        self.doubleArrayNode = None
        self.spectrumImageNode = None
        self.observerTags = []
        self.outputArrayNode = None

        # Spectrum Analysis Variables
        self.spectraCollected = 0
        self.averageSpectrumDifferences = 0
        self.numberOfSpectrumDataPoints = 100
        self.firstComparison = 0
        self.currentSpectrum = vtk.vtkPoints()
        self.referenceSpectra = vtk.vtkPolyData()
        self.spectra = vtk.vtkPoints()

        # Cooridinate Variables
        self.xcoordinate = 0
        self.ycoordinate = 0
        self.zcoordinate = 0
        self.genFidIndex = 0
        self.regFidIndex = 0
        self.boundFidIndex = 0

        # Contour Tracing Variables
        self.pointsForHull = vtk.vtkPoints()
        self.firstDataPointGenerated = 0
        self.edgePoint = 0
        self.pointsForEdgeTracing = vtk.vtkPoints()
        self.edgeTracingTimerStart = 2000
        self.createTumorArray = 0
        self.startNext = 6000
        self.timerTracker = 0

        # General Movement Variables
        self.fiducialMovementDelay = 0
        self.currentXcoordinate = 0
        self.currentYcoordinate = 0

        # instantiate coordinate values
        self.getCoordinateCmd = slicer.vtkSlicerOpenIGTLinkCommand()
        self.getCoordinateCmd.SetCommandName('SendText')
        self.getCoordinateCmd.SetCommandAttribute('DeviceId', "SerialDevice")
        self.getCoordinateCmd.SetCommandTimeoutSec(1.0)
        self.getCoordinateCmd.SetCommandAttribute('Text', 'M114')
        self.getCoordinateCmd.AddObserver(self.getCoordinateCmd.CommandCompletedEvent, self.onPrinterCommandCompleted)
        # instantiate landmark coordinate command
        self.landmarkCoordinateCmd = slicer.vtkSlicerOpenIGTLinkCommand()
        self.landmarkCoordinateCmd.SetCommandName('SendText')
        self.landmarkCoordinateCmd.SetCommandAttribute('DeviceId', "SerialDevice")
        self.landmarkCoordinateCmd.SetCommandTimeoutSec(1.0)
        self.landmarkCoordinateCmd.SetCommandAttribute('Text', 'M114')
        self.landmarkCoordinateCmd.AddObserver(self.getCoordinateCmd.CommandCompletedEvent,self.onLandmarkCoordinateCmd)
        # instantiate boundary coordinate command
        self.boundaryCoordinateCmd = slicer.vtkSlicerOpenIGTLinkCommand()
        self.boundaryCoordinateCmd.SetCommandName('SendText')
        self.boundaryCoordinateCmd.SetCommandAttribute('DeviceId', "SerialDevice")
        self.boundaryCoordinateCmd.SetCommandTimeoutSec(1.0)
        self.boundaryCoordinateCmd.SetCommandAttribute('Text', 'M114')
        self.boundaryCoordinateCmd.AddObserver(self.boundaryCoordinateCmd.CommandCompletedEvent,self.onBoundaryCoordinateCmd)
        # instantiate home command
        self.homeCmd = slicer.vtkSlicerOpenIGTLinkCommand()
        self.homeCmd.SetCommandName('SendText')
        self.homeCmd.SetCommandAttribute('DeviceId', "SerialDevice")
        self.homeCmd.SetCommandTimeoutSec(1.0)
        self.homeCmd.SetCommandAttribute('Text', 'G28 X Y ')
        # instantiate emergency stop command
        self.emergStopCmd = slicer.vtkSlicerOpenIGTLinkCommand()
        self.emergStopCmd.SetCommandName('SendText')
        self.emergStopCmd.SetCommandAttribute('DeviceId', "SerialDevice")
        self.emergStopCmd.SetCommandTimeoutSec(1.0)
        self.emergStopCmd.SetCommandAttribute('Text', 'M112')
        # instantiate x command
        self.xControlCmd = slicer.vtkSlicerOpenIGTLinkCommand()
        self.xControlCmd.SetCommandName('SendText')
        self.xControlCmd.SetCommandAttribute('DeviceId', "SerialDevice")
        self.xControlCmd.SetCommandTimeoutSec(1.0)
        # instantiate y command
        self.yControlCmd = slicer.vtkSlicerOpenIGTLinkCommand()
        self.yControlCmd.SetCommandName('SendText')
        self.yControlCmd.SetCommandAttribute('DeviceId', "SerialDevice")
        self.yControlCmd.SetCommandTimeoutSec(1.0)
        # instantiate move middle command
        self.printerControlCmd = slicer.vtkSlicerOpenIGTLinkCommand()
        self.printerControlCmd.SetCommandName('SendText')
        self.printerControlCmd.SetCommandAttribute('DeviceId', "SerialDevice")
        self.printerControlCmd.SetCommandTimeoutSec(1.0)
        # instantiate move X and Y command
        self.xyControlCmd = slicer.vtkSlicerOpenIGTLinkCommand()
        self.xyControlCmd.SetCommandName('SendText')
        self.xyControlCmd.SetCommandAttribute('DeviceId', "SerialDevice")
        self.xyControlCmd.SetCommandTimeoutSec(1.0)
        # instantiate move Z command
        self.zControlCmd = slicer.vtkSlicerOpenIGTLinkCommand()
        self.zControlCmd.SetCommandName('SendText')
        self.zControlCmd.SetCommandAttribute('DeviceId', "SerialDevice")
        self.zControlCmd.SetCommandTimeoutSec(1.0)

    def setSerialIGTLNode(self, serialIGTLNode):
        self.serialIGTLNode = serialIGTLNode

    def setdoubleArrayNode(self, doubleArrayNode):
        self.doubleArrayNode = doubleArrayNode

    def addObservers(self):
        if self.spectrumImageNode:
            print "Add observer to {0}".format(self.spectrumImageNode.GetName())
            self.observerTags.append([self.spectrumImageNode,self.spectrumImageNode.AddObserver(vtk.vtkCommand.ModifiedEvent,self.onSpectrumImageNodeModified)])

    def removeObservers(self):
        print "Remove observers"
        for nodeTagPair in self.observerTags:
            nodeTagPair[0].RemoveObserver(nodeTagPair[1])

    def onSpectrumImageNodeModified(self, observer, eventid):
        if not self.spectrumImageNode or not self.outputArrayNode:
            return
        self.updateOutputArray()
        self.updateChart()

    def updateOutputArray(self, node):
        self.spectrumImageNode = node
        numberOfPoints = self.spectrumImageNode.GetImageData().GetDimensions()[0]
        numberOfRows = self.spectrumImageNode.GetImageData().GetDimensions()[1]
        if numberOfRows != 2:
            logging.error("Spectrum image is expected to have exactly 2 rows, got {0}".format(numberOfRows))
            return

        # Create arrays of data
        a = self.outputArrayNode.GetArray()
        a.SetNumberOfTuples(self.numberOfSpectrumDataPoints)

        for row in xrange(numberOfRows):
            lineSource = vtk.vtkLineSource()
            lineSource.SetPoint1(0, row, 0)
            lineSource.SetPoint2(numberOfPoints - 1, row, 0)
            lineSource.SetResolution(self.numberOfSpectrumDataPoints - 1)
            probeFilter = vtk.vtkProbeFilter()
            probeFilter.SetInputConnection(lineSource.GetOutputPort())
            if vtk.VTK_MAJOR_VERSION <= 5:
                probeFilter.SetSource(self.spectrumImageNode.GetImageData())
            else:
                probeFilter.SetSourceData(self.spectrumImageNode.GetImageData())
            probeFilter.Update()
            probedPoints = probeFilter.GetOutput()
            probedPointScalars = probedPoints.GetPointData().GetScalars()
            for i in xrange(self.numberOfSpectrumDataPoints):
                a.SetComponent(i, row, probedPointScalars.GetTuple(i)[0])

        for i in xrange(self.numberOfSpectrumDataPoints):
            a.SetComponent(i, 2, 0)

        probedPoints.GetPointData().GetScalars().Modified()

    # These two functions offer the same functionality as xrange but are able to accept floating point values. Implemented to facilitate high resolution scanning.

    def frange(self, start, end, stepsize):
        while start < end:
            yield start
            start += stepsize

    def backfrange(self, start, end, stepsize):
        while start > end:
            yield start
            start += stepsize

    def getSpectralData(self, outputArrayNode):
        self.referenceOutputArrayNode = outputArrayNode
        referencePointsArray = self.referenceOutputArrayNode.GetArray()

        self.spectra.SetNumberOfPoints(100)
        for i in xrange(0, 101, 1):
            self.spectra.SetPoint(i, referencePointsArray.GetTuple(i))

        self.spectraCollected = 1
        print"Spectra collected."

    def home(self):
        slicer.modules.openigtlinkremote.logic().SendCommand(self.homeCmd, self.serialIGTLNode.GetID())

                                                            # Keyboard Shortcuts
    # Keyboard shortcuts allow the user to manipulate the printer bed with the arrow keys, implemented for boundary selection in ROI scanning.

    def declareShortcut(self, serialIGTLNode):
        # necessary to have this in a function activated by Active keyboard short cut function so that the movements can be instantiated after IGTL has already been instantiated.
        self.installShortcutKeys(serialIGTLNode)

    def installShortcutKeys(self, serialIGTLNode):
        self.shortcuts = []
        keysAndCallbacks = (
            ('Right', lambda: self.keyboardControlledXMovementForward(serialIGTLNode)),
            ('Left', lambda: self.keyboardControlledXMovementBackwards(serialIGTLNode)),
            ('Up', lambda: self.keyboardControlledYMovementForward(serialIGTLNode)),
            ('Down', lambda: self.keyboardControlledYMovementBackwards(serialIGTLNode)),
            ('H', lambda: self.keyboardControlledHomeMovement(serialIGTLNode)),
        )

        for key, callback in keysAndCallbacks:
            shortcut = qt.QShortcut(slicer.util.mainWindow())
            shortcut.setKey(qt.QKeySequence(key))
            shortcut.connect('activated()', callback)
            self.shortcuts.append(shortcut)

                                                            # Locational Information
            
    # These functions are used to access the coordinate location of the fiber probe and communicate that to slicer for application in the slicer 3D scene. 

    def parseCoords(self, mylist):
        # Parse string for x coordinate value
        xvalues = mylist[0].split(":")
        self.xcoordinate = float(xvalues[1])

        # Parse string for y coordinate value
        yvalues = mylist[1].split(":")
        self.ycoordinate = float(yvalues[1])

        # Parse string for z coordinate value
        zvalues = mylist[2].split(":")
        self.zcoordinate = float(zvalues[1])

        return self.xcoordinate, self.ycoordinate, self.zcoordinate

    def get_coordinates(self):
        slicer.modules.openigtlinkremote.logic().SendCommand(self.getCoordinateCmd, self.serialIGTLNode.GetID())
        return self.xcoordinate, self.ycoordinate

    def onPrinterCommandCompleted(self, observer, eventid):
        coordinateValues = self.getCoordinateCmd.GetResponseMessage()
        print("Command completed with status: " + self.getCoordinateCmd.StatusToString(
            self.getCoordinateCmd.GetStatus()))
        print("Response message: " + coordinateValues)
        print("Full response: " + self.getCoordinateCmd.GetResponseText())

        mylist = coordinateValues.split(" ")
        self.xcoordinate, self.ycoordinate, self.zcoordinate = self.parseCoords(mylist)

        # for automated edge tracing
        if self.edgePoint == 0:
            self._savexcoordinate.append(self.xcoordinate)
            self._saveycoordinate.append(self.ycoordinate)
            self.edgePoint = 1

        self.dataCollection = self.createPolyDataPoint(self.xcoordinate, self.ycoordinate, self.zcoordinate)

        if self.genFidIndex< 1:
            self.fiducialMarker(self.xcoordinate, self.ycoordinate + 1, self.zcoordinate)
            self.genFidIndex= self.genFidIndex + 1
            # Only creates ONE node
        elif self.genFidIndex== 1234:
            return self.xcoordinate
            # for turning fiducial marking off
        else:
            self.addToCurrentFiducialNode(self.xcoordinate, self.ycoordinate + 1, self.zcoordinate)

        return self.xcoordinate, self.ycoordinate

    def fiducialMarkerChecked(self):
        self.genFidIndex= 1234  # will break if 1234 fiducials is ever reached, implemented for the fiducial marking off function

    def fiducialMarker(self, xcoordinate, ycoordinate, zcoordinate):
        self.fiducialNode = slicer.vtkMRMLMarkupsFiducialNode()
        slicer.mrmlScene.AddNode(self.fiducialNode)
        self.fiducialNode.SetName("")
        self.fiducialNode.SetNthFiducialLabel(0, "")
        self.fiducialNode.AddFiducial(xcoordinate, ycoordinate, zcoordinate)

    def addToCurrentFiducialNode(self, xcoordinate, ycoordinate, zcoordinate):
        self.fiducialNode.AddFiducial(xcoordinate, ycoordinate, zcoordinate)
        self.fiducialNode.SetNthFiducialLabel(self.genFidIndex, "")
        self.genFidIndex= self.genFidIndex + 1


                                                    # Spectrum Comparison
    
    # Spectrum comparison is used to determine where the live spectrum is the same as the reference spectrum collected before scanning. 
    
    def spectrumComparison(self, outputArrayNode):
        
        if self.spectraCollected == 0:
            print " Error: reference spectrum not collected."
            return

        self.currentOutputArrayNode = outputArrayNode
        currentPointsArray = self.currentOutputArrayNode.GetArray()
        # Data is acquired from probe in a double array with each index corresponding to either wavelength or intensity
        # There are 100 points (tuples) each consisting of one wavelength and a corresponding intensity
        # The first index (0) is where wavelength values are stored
        # The second index (1) is where intensities are stored
        self.currentSpectrum.SetNumberOfPoints(100)
        for i in xrange(0, 101, 1):
            self.currentSpectrum.SetPoint(i, currentPointsArray.GetTuple(i))

        self.averageSpectrumDifferences = 0

        for j in xrange(0, 101, 1):
            x = self.currentSpectrum.GetPoint(j)
            y = self.spectra.GetPoint(j)
            self.averageSpectrumDifferences = self.averageSpectrumDifferences + (y[1] - x[1])


        if abs(self.averageSpectrumDifferences) < 10: # 10 is the threshold
            print " tumor"
            if self.firstComparison == 1:
                self.get_coordinates()
            if self.createTumorArray == 1:
                self._tumorCheck.append(1)
            return False
        else:
            print "healthy"
            if self.createTumorArray == 1:
                self._tumorCheck.append(0)
            return True


                                                                # Systematic Scanning Scheme

    # Systematic scanning is facilitated by controlled x and y scanning oscillations. X oscilattions forward and backwards are regulated by the X-Loop function and called at the same time.
    # Y movement is only implemented in one direction (backwards) and is called after each individual oscillation (forward and backward).
    # Delays for scanning movement are developed to execute incremental delays, the mathematical expressions are dependent on the loop iterator.
    # Systematic scan occurs in a rectangular grid pattern

    def yLoop(self, mvmtDelay, yResolution, xResolution):
        i = 0
        j = 0
        printerBedWidth = (120 / xResolution)
        fullOscillationWidth = 2 * (120 / xResolution) 

        # call the first two y Movements (helps with oscillation delays)
        self.yMovement((printerBedWidth + 2) * mvmtDelay, yResolution)
        self.yMovement((fullOscillationWidth + 1) * mvmtDelay, yResolution * 2)
       
        if yResolution < 38 or yResolution == 40: 
            for yValue in self.frange(yResolution * 3, 120 + yResolution, yResolution * 2):  
                delayMs = (((fullOscillationWidth + printerBedWidth) + 2) * mvmtDelay) + ((fullOscillationWidth * mvmtDelay) * i)
                self.yMovement(delayMs, yValue)
                i = i + 1
            for yValue in self.frange(yResolution * 4, 120 + yResolution, yResolution * 2):
                delayMs = ((((2 * fullOscillationWidth) + 1)) * mvmtDelay) + ((fullOscillationWidth * mvmtDelay) * j)
                self.yMovement(delayMs, yValue)
                j = j + 1
        else:
            for yValue in self.frange(yResolution * 3, 120, yResolution * 2):
                delayMs = (((fullOscillationWidth + printerBedWidth) + 2) * mvmtDelay) + ((fullOscillationWidth * mvmtDelay) * i)
                self.yMovement(delayMs, yValue)
                i = i + 1
            for yValue in self.frange(yResolution * 4, 120, yResolution * 2):
                delayMs = ((((2 * fullOscillationWidth) + 1)) * mvmtDelay) + ((fullOscillationWidth * mvmtDelay) * j)
                self.yMovement(delayMs, yValue)
                j = j + 1

    def xLoop(self, mvmtDelay, xResolution, yResolution):
        oscillatingTime = (120 / yResolution) / 2  
        xOscillation = ((120 / xResolution) * 2) * mvmtDelay  
        for xCoordinateValue in self.frange(0, (oscillatingTime * xOscillation) + (xOscillation), xOscillation):
            self.xWidthForward(xCoordinateValue, mvmtDelay, xResolution)
            self.xWidthBackwards(xCoordinateValue, mvmtDelay, xResolution)

    def xWidthForward(self, xCoordinate, mvmtDelay, xResolution):
        if xResolution < 38 or xResolution == 40:
            for xValue in self.frange(0, 120 + xResolution, xResolution):
                delayMs = xCoordinate + xValue * (mvmtDelay / xResolution)
                self.XMovement(delayMs, xValue)
        else:
            for xValue in self.frange(0, 120, xResolution):
                delayMs = xCoordinate + xValue * (mvmtDelay / xResolution)
                self.XMovement(delayMs, xValue)

    def xWidthBackwards(self, xCoordinate, mvmtDelay, xResolution):
        if xResolution < 38 or xResolution == 40:
            for xValue in self.backfrange(120, -xResolution, -xResolution):
                delayMs = abs(xValue - 120) * (mvmtDelay / xResolution) + (120 / xResolution + 1) * mvmtDelay + xCoordinate
                self.XMovement(delayMs, xValue)
        else:
            for xValue in self.backfrange(120, 0, -xResolution):
                delayMs = abs(xValue - 120) * (mvmtDelay / xResolution) + (120 / xResolution + 1) * mvmtDelay + xCoordinate
                self.XMovement(delayMs, xValue)

                                                # ROI Systematic Searching

    # ROI Systematic Scanning is implemented for controlled, high resolution systematic scanning.
    # Boundaries are selected by the user using the keyboard control buttons and physical location of the probe.
    # From the boundaries, the ROI is selected and a systematic grid scan is executed.

    def getBoundaryFiducialsCoordinate(self):
        slicer.modules.openigtlinkremote.logic().SendCommand(self.boundaryCoordinateCmd, self.serialIGTLNode.GetID())

    def onBoundaryCoordinateCmd(self, observer, eventid):
        coordinateValues = self.boundaryCoordinateCmd.GetResponseMessage()
        print("Command completed with status: " + self.boundaryCoordinateCmd.StatusToString(
            self.boundaryCoordinateCmd.GetStatus()))
        print("Response message: " + coordinateValues)
        print("Full response: " + self.boundaryCoordinateCmd.GetResponseText())
        # parsing the string for specific coordinate values
        mylist = coordinateValues.split(" ")

        self.xcoordinate, self.ycoordinate, self.zcoordinate = self.parseCoords(mylist)

        if self.boundFidIndex < 1:
            self.boundaryFiducialMarker(self.xcoordinate, self.ycoordinate, self.zcoordinate)
            self.boundFidIndex = self.boundFidIndex + 1
        elif self.boundFidIndex == 1234:
            return self.xcoordinate
        else:
            self.addtoBoundaryFiducialMarker(self.xcoordinate, self.ycoordinate, self.zcoordinate)

    def boundaryFiducialMarker(self, xcoordinate, ycoordinate, zcoordinate):
        self.fiducialNode2 = slicer.vtkMRMLMarkupsFiducialNode()
        self.fiducialNode2.SetName("BoundaryPoints")
        slicer.mrmlScene.AddNode(self.fiducialNode2)
        self.fiducialNode2.AddFiducial(xcoordinate, ycoordinate, zcoordinate)

    def addtoBoundaryFiducialMarker(self, xcoordinate, ycoordinate, zcoordinate):
        self.fiducialNode2.AddFiducial(xcoordinate, ycoordinate, zcoordinate)
        self.boundFidIndex = self.boundFidIndex + 1

    def ROIBoundarySearch(self):
        ILfidList = slicer.util.getNode('BoundaryPoints')
        if not ILfidList:
            print "Error: No boundary points selected."
            return False
        else:
            numFids = ILfidList.GetNumberOfFiducials()

            for i in xrange(numFids):
                ras = [0, 0, 0]
                pos = ILfidList.GetNthFiducialPosition(i, ras)
                world = [0, 0, 0, 0]
                ILfidList.GetNthFiducialWorldCoordinates(0, world)
                xcoord = int(ras[0])
                ycoord = int(ras[1])
                self._ROIxbounds.append(xcoord)
                self._ROIybounds.append(ycoord)

            xMin = min(self._ROIxbounds)
            xMax = max(self._ROIxbounds)
            yMin = min(self._ROIybounds)
            yMax = max(self._ROIybounds)
            return xMin, xMax, yMin, yMax

    def ROIsearchXLoop(self, mvmtDelay, xResolution, yResolution, xMin, xMax, yMin, yMax):

        oscillatingTime = ((yMax - yMin) / yResolution) / 2
        xOscillation = (((xMax - xMin) / xResolution) * 2) * mvmtDelay
            
        for xCoordinateValue in self.frange(0, xOscillation * oscillatingTime,xOscillation):
            self.ROIsearchXWidthForward(xCoordinateValue, mvmtDelay, xResolution, xMin, xMax)
            self.ROIsearchXWidthBackward(xCoordinateValue, mvmtDelay, xResolution, xMin, xMax)

    def ROIsearchXWidthForward(self, xCoordinate, mvmtDelay, xResolution, xMin, xMax):

        for xValue in self.frange(xMin, xMax, xResolution):
            delayMs = xCoordinate + (xValue - xMin) * (mvmtDelay / xResolution) + mvmtDelay
            self.XMovement(delayMs, xValue)

    def ROIsearchXWidthBackward(self, xCoordinate, mvmtDelay, xResolution, xMin, xMax):

        for xValue in self.backfrange(xMax, xMin, -xResolution):
            delayMs = xCoordinate + (((xMax - xMin) / xResolution) * mvmtDelay) + (xMax - xValue) * (mvmtDelay / abs(xResolution)) + mvmtDelay
            self.XMovement(delayMs, xValue)

    def ROIsearchYLoop(self, mvmtDelay, yResolution, xResolution, yMin, yMax, xMin, xMax):

        i = 0
        j = 0
        printerBedWidth = ((xMax - xMin) / xResolution)
        fullOscillationWidth = 2 * ((xMax - xMin) / xResolution)
        self.yMovement((printerBedWidth + 2) * mvmtDelay, yResolution + yMin)
        self.yMovement((fullOscillationWidth + 1) * mvmtDelay, (yResolution * 2) + yMin)

        for yValue in self.frange(yMin + (yResolution * 3), yMax + yResolution, yResolution * 2):
            delayMs = (((fullOscillationWidth + printerBedWidth) + 2) * mvmtDelay) + ((fullOscillationWidth * mvmtDelay) * i)
            self.yMovement(delayMs, yValue)
            i = i + 1
        for yValue in self.frange(yMin + (yResolution * 4), yMax + yResolution, yResolution * 2):
            delayMs = ((((2 * fullOscillationWidth) + 1)) * mvmtDelay) + ((fullOscillationWidth * mvmtDelay) * j)
            self.yMovement(delayMs, yValue)
            j = j + 1


                    # ROI Rasterization Scanning
        # This function initiates a zig-zag raster pattern within a ROI, delivers more accurate scanning that systematic rectilinear scan

    def calldiagonalforward(self, xResolution, yResolution, timeDelay, ddMs, b):
        dTimer = qt.QTimer()
        dTimer.singleShot(timeDelay, lambda: self.diagonalforward(xResolution,yResolution,ddMs, b))

    def calldiagonalbackward(self, xResolution, yResolution, timeDelay, ddMs,b):
        dTimer = qt.QTimer()
        dTimer.singleShot(timeDelay, lambda: self.diagonalbackward(xResolution,yResolution,ddMs, b))

    def diagonalforward(self,  xResolution, yResolution, timeDelay,b):

        xMin, xMax, yMin, yMax = self.ROIBoundarySearch()
        deltaY = (yResolution)
        deltaX = xMax - xMin
        slope = deltaY/deltaX

        self.i = 0
        for x in self.frange(xMin,xMax,xResolution):

            delayMs = timeDelay * self.i
            y = slope*x + b
            self.xyMovement(x,y,delayMs)
            self.i = self.i + 1

    def diagonalbackward(self,xResolution, yResolution, timeDelay, b):
        xMin, xMax, yMin, yMax = self.ROIBoundarySearch()
        deltaY = (yResolution)
        deltaX = xMax - xMin
        slope = -(deltaY / deltaX)

        self.j = 0
        for x in self.backfrange(xMax, xMin, -xResolution):
            delayMs = timeDelay * self.j

            y = slope * x + b
            self.xyMovement(x, y, delayMs)
            self.j = self.j + 1


    def contour_pattern(self):
        pattern = np.load("C:\Users\lconnolly\Desktop\use_this_tissue_scanning/transformed pts.npy")
        
        timevar = 0
        for j in range(0,len(pattern)):
            x = pattern[j][0]
            y = pattern[j][1]
            self.xyMovement(x,y, timevar)
            timevar = timevar + 1000

    def zigzagPattern(self, xResolution, yResolution,  ddMs):

        xMin, xMax, yMin, yMax = self.ROIBoundarySearch()
        delayX = (xMax-xMin)/ xResolution * ddMs
        delayY = (yMax-yMin)/ yResolution * delayX

        self.k = 1
        self.l = 2

        for callDelay in self.frange(0,delayY, delayX*2):
            print callDelay, callDelay + delayX
            bfwd = yMin + (self.k*yResolution)
            bbkwd = yMin + (self.l*yResolution)
            self.calldiagonalforward(xResolution, yResolution, callDelay, ddMs, bfwd)
            self.calldiagonalbackward(xResolution, yResolution, callDelay+delayX, ddMs, bbkwd)
            self.k = self.k + 2
            self.l = self.l + 2


                                                # Contour Tracing - After Systematic Scan

    # The following code was developed for contour tracing following a systamtic scan by determining the convex hull of a list of collected points. Each coordinate point is
    # saved in a polydata point in the get_coordinates function. After the scan, if the user selects contour trace, the z axis is lowered 5 mm and the probe with then move to trace the
    #convex hull of the previously collected data points.

    def createPolyDataPoint(self, xcoordinate, ycoordinate, zcoordinate):
        if self.firstDataPointGenerated < 1:
            self.firstDataPointGenerated = self.firstDataPointGenerated + 1
            self.pointsForHull.InsertNextPoint(xcoordinate, ycoordinate, zcoordinate)
        else:
            self.pointsForHull.InsertNextPoint(xcoordinate, ycoordinate, zcoordinate)

    def convexHull(self):

        self.hullPolydata = vtk.vtkPolyData()
        self.hullPolydata.SetPoints(self.pointsForHull)

        hull = vtk.vtkConvexHull2D()
        hull.SetInputData(self.hullPolydata)
        hull.Update()

        pointLimit = hull.GetOutput().GetNumberOfPoints()
        for i in xrange(0, pointLimit):
            self.pointsForEdgeTracing.InsertNextPoint(hull.GetOutput().GetPoint(i))
        self.getCoordinatesForEdgeTracing(self.pointsForEdgeTracing, pointLimit)

    def getCoordinatesForEdgeTracing(self, pointsForEdgeTracing, pointLimit):

        for i in xrange(0, pointLimit):
            pointVal = pointsForEdgeTracing.GetPoint(i)
            xcoordinate = pointVal[0]
            ycoordinate = pointVal[1]
            self._xHullArray.append(xcoordinate)
            self._yHullArray.append(ycoordinate)
            self.slowEdgeTracing(xcoordinate, ycoordinate, self.edgeTracingTimerStart)

            self.edgeTracingTimerStart = self.edgeTracingTimerStart + 2000
        self.slowEdgeTracing(self._xHullArray[0], self._yHullArray[0], (2000 * pointLimit + 2000))
        self.ZMovement(2000, -5)
        self.ZMovement(2000 * pointLimit + 4000, 0)

                                        # Contour Tracing - Independent of Systematic Scan

# This code was developed to facilitate automated edge tracing without any systematic scanning. The probe moves in at the specified resolution in a +x,-y,-x,+y pattern iteratively
# and determines it's new trajectory based on the spectrum in each quadrant.

    def callNewOrigin(self, delay):
        originTimer = qt.QTimer()
        originTimer.singleShot(delay, lambda: self.newOrigin())

    def callQuadrantCheck(self, delay, outputArrayNode, quadrantResolution):
        quadTimer = qt.QTimer()
        quadTimer.singleShot(delay, lambda: self.checkQuadrantValues(outputArrayNode, quadrantResolution))

    def callMovement(self, delay, xcoordinate, ycoordinate):
        self.cutInTimer = qt.QTimer()
        self.cutInTimer.singleShot(delay, lambda: self.controlledXYMovement(xcoordinate, ycoordinate))

    def callGetCoordinates(self, delay):
        coordTimer = qt.QTimer()
        coordTimer.singleShot(delay, lambda: self.get_coordinates())

    def call_getCoordinates(self, delay):
        edgeTraceTimer = qt.QTimer()
        edgeTraceTimer.singleShot(delay, lambda: self.get_coordinates())

    def readCoordinatesAtTimeInterval(self, delay, outputArrayNode):
        self.firstComparison = 1
        self.edgeTraceTimer.singleShot(delay, lambda: self.spectrumComparison(outputArrayNode))

    def readCoordinatesAtTimeInterval2(self, delay, outputArrayNode):
        self.createTumorArray = 1
        self.edgeTraceTimer.singleShot(delay, lambda: self.spectrumComparison(outputArrayNode))

    def readCoordinatesAtTimeInterval3(self, delay, outputArrayNode):
        self.edgeTraceTimer.singleShot(delay, lambda: self.spectrumComparison(outputArrayNode))

    def moveBackToOriginalEdgePoint(self, lastdelay):
        x = len(self._savexcoordinate) - 1
        self.edgeTraceTimer.singleShot(lastdelay, lambda: self.controlledXYMovement(self._savexcoordinate[x],
                                                                                        self._saveycoordinate[x]))

    def edgeTrace(self, outputArrayNode, quadrantResolution):
        qt = (5 * 1000) + 4000
        for i in self.frange(0, 100000, qt):
            self.callQuadrantCheck(i, outputArrayNode, quadrantResolution)
            self.callGetCoordinates(i + 7000)
            self.callNewOrigin(i + 7500)

    def findAndMoveToEdge(self, outputArrayNode):
        xMin, xMax, yMin, yMax = self.ROIBoundarySearch()
        self.edgeTraceTimer = qt.QTimer()
        self.callMovement(0,xMin,yMin)

        for y in xrange(xMin, xMax + 1, 1):
            delayMs = ((y / 2) * 500) - 6000
            self.callMovement(delayMs, y, yMin)
            yMin = yMin + 1

        lastdelay = delayMs + 1000

        for x in xrange(0, lastdelay, 500):
            self.readCoordinatesAtTimeInterval(x, outputArrayNode)

        self.moveBackToOriginalEdgePoint(lastdelay)

    def checkQuadrantValues(self, outputArrayNode, quadrantResolution):
            # go right, back, left, forward until you determine which quadrant to continue in
        self.printTimer = qt.QTimer()
        index = len(self._savexcoordinate) - 1

        self.callMovement(1000, (self._savexcoordinate[index] + quadrantResolution), (self._saveycoordinate[index]))
        self.readCoordinatesAtTimeInterval2(2000, outputArrayNode)

        self.callMovement(2000, (self._savexcoordinate[index]), (self._saveycoordinate[index] - quadrantResolution))
        self.readCoordinatesAtTimeInterval2(3000, outputArrayNode)

        self.callMovement(3000, (self._savexcoordinate[index] - quadrantResolution), (self._saveycoordinate[index]))
        self.readCoordinatesAtTimeInterval2(4000, outputArrayNode)

        self.callMovement(4000, (self._savexcoordinate[index]), (self._saveycoordinate[index] + quadrantResolution))
        self.readCoordinatesAtTimeInterval2(5000, outputArrayNode)

        self.callMovement(5000, (self._savexcoordinate[index]), (self._saveycoordinate[index]))

        self.startTrajectorySearch(outputArrayNode, quadrantResolution)
        self.timerTracker = self.timerTracker + 6000

    def findTrajectory(self, outputArrayNode, quadrantResolution):
        self.trajectoryTimer = qt.QTimer()
        index = len(self._tumorCheck)
        y = len(self._savexcoordinate) - 1

        if (self._tumorCheck[index - 4] == 1 and self._tumorCheck[index - 3] == 0 and self._tumorCheck[
                index - 2] == 0 and self._tumorCheck[index - 1] == 1) or ((
                    self._tumorCheck[index - 4] == 0 and self._tumorCheck[index - 3] == 0 and self._tumorCheck[
                index - 2] == 0 and self._tumorCheck[index - 1] == 1) or (self._tumorCheck[index - 4] == 1 and self._tumorCheck[index - 3] == 1 and self._tumorCheck[
                index - 2] == 0 and self._tumorCheck[index - 1] == 1)):
            print "Quadrant 2"
            self.callMovement(0, self._savexcoordinate[y] - (quadrantResolution - 1),
                                  self._saveycoordinate[y] + (quadrantResolution + 1))

        if (self._tumorCheck[index - 4] == 1 and self._tumorCheck[index - 3] == 1 and self._tumorCheck[
                index - 2] == 0 and self._tumorCheck[index - 1] == 0) or (
                    self._tumorCheck[index - 4] == 1 and self._tumorCheck[index - 3] == 0 and self._tumorCheck[
                index - 2] == 0 and self._tumorCheck[index - 1] == 0) or ((
                    self._tumorCheck[index - 4] == 1 and self._tumorCheck[index - 3] == 1 and self._tumorCheck[
                index - 2] == 1 and self._tumorCheck[index - 1] == 0)) or ((
                    self._tumorCheck[index - 4] == 1 and self._tumorCheck[index - 3] == 1 and self._tumorCheck[
                index - 2] == 1 and self._tumorCheck[index - 1] == 1)):
            print "Quadrant 1"
            self.callMovement(0, self._savexcoordinate[y] + (quadrantResolution - 1),
                                  self._saveycoordinate[y] + (quadrantResolution + 1))

        if (self._tumorCheck[index - 4] == 0 and self._tumorCheck[index - 3] == 0 and self._tumorCheck[
                index - 2] == 1 and self._tumorCheck[index - 1] == 1) or (
                    self._tumorCheck[index - 4] == 0 and self._tumorCheck[index - 3] == 0 and self._tumorCheck[
                index - 2] == 1 and self._tumorCheck[index - 1] == 0) or ((
                    self._tumorCheck[index - 4] == 1 and self._tumorCheck[index - 3] == 1 and self._tumorCheck[
                index - 2] == 1 and self._tumorCheck[index - 1] == 1)) or ((
                    self._tumorCheck[index - 4] == 1 and self._tumorCheck[index - 3] == 0 and self._tumorCheck[
                index - 2] == 1 and self._tumorCheck[index - 1] == 1)):
            print "Quadrant 3"
            self.callMovement(0, self._savexcoordinate[y] - (quadrantResolution - 1),
                                  self._saveycoordinate[y] - (quadrantResolution + 1))

        if (self._tumorCheck[index - 4] == 0 and self._tumorCheck[index - 3] == 1 and self._tumorCheck[
                index - 2] == 1 and self._tumorCheck[index - 1] == 0) or (
                    self._tumorCheck[index - 4] == 0 and self._tumorCheck[index - 3] == 1 and self._tumorCheck[
                index - 2] == 0 and self._tumorCheck[index - 1] == 0) or ((
                    self._tumorCheck[index - 4] == 0 and self._tumorCheck[index - 3] == 1 and self._tumorCheck[
                index - 2] == 1 and self._tumorCheck[index - 1] == 1)) or (self._tumorCheck[index - 4] == 0 and self._tumorCheck[index - 3] == 0 and self._tumorCheck[
                index - 2] == 0 and self._tumorCheck[index - 1] == 0):
            print "Quadrant 4"
            self.callMovement(0, self._savexcoordinate[y] + (quadrantResolution - 1),
                                  self._saveycoordinate[y] - (quadrantResolution + 1))

        if (self._tumorCheck[index - 4] == 0 and self._tumorCheck[index - 3] == 0 and self._tumorCheck[
                index - 2] == 0 and self._tumorCheck[index - 1] == 0):
            self.offSpecimen = 1

            self.edgePoint = 0
            self.call_getCoordinates(self.timerTracker + 1000)

    def newOrigin(self):
        self.edgePoint = 0
        self.get_coordinates()
        if self.genFidIndex < 1:
            self.fiducialMarker(self.xcoordinate, self.ycoordinate, self.zcoordinate)
            self.genFidIndex = self.genFidIndex + 1
        else:
            self.addToCurrentFiducialNode(self.xcoordinate, self.ycoordinate, self.zcoordinate)

    def startTrajectorySearch(self, outputArrayNode, quadrantResolution):
        trajTimer = qt.QTimer()
        trajTimer.singleShot(self.startNext,lambda: self.findTrajectory(outputArrayNode, quadrantResolution))  # was self.startNext

                                                    # Image Registration Tools

    # For image registration, the user must indicate where the registration points are using either the arrow keys or manually select the points in the slicer window.
    # Landmark registration is then used to compute the transform and to visualize the registration, the user must organize the transform hierarchy properly in slicer
    # and check volume reslice driver.

    def getLandmarkFiducialsCoordinate(self):
        slicer.modules.openigtlinkremote.logic().SendCommand(self.landmarkCoordinateCmd, self.serialIGTLNode.GetID())

    def onLandmarkCoordinateCmd(self, observer, eventid):
        coordinateValues = self.landmarkCoordinateCmd.GetResponseMessage()
        print("Command completed with status: " + self.landmarkCoordinateCmd.StatusToString(
            self.landmarkCoordinateCmd.GetStatus()))
        print("Response message: " + coordinateValues)
        print("Full response: " + self.landmarkCoordinateCmd.GetResponseText())

        mylist = coordinateValues.split(" ")
        self.xcoordinate, self.ycoordinate, self.zcoordinate = self.parseCoords(mylist)

        if self.regFidIndex < 1:
            self.landmarkFiducialMarker(self.xcoordinate, self.ycoordinate, self.zcoordinate)
            self.regFidIndex = self.regFidIndex + 1
        elif self.genFidIndex== 1234:
            return self.xcoordinate
        else:
            self.addToLandmarkFiducialNode(self.xcoordinate, self.ycoordinate, self.zcoordinate)

    def landmarkFiducialMarker(self, xcoordinate, ycoordinate, zcoordinate):
        self.fiducialNode1 = slicer.vtkMRMLMarkupsFiducialNode()
        self.fiducialNode1.SetName("ModelLandmarkPoints")
        slicer.mrmlScene.AddNode(self.fiducialNode1)
        self.fiducialNode1.AddFiducial(xcoordinate, ycoordinate, zcoordinate)

    def addToLandmarkFiducialNode(self, xcoordinate, ycoordinate, zcoordinate):
        self.fiducialNode1.AddFiducial(xcoordinate, ycoordinate, zcoordinate)

    def ICPRegistration(self):
        ILfidList = slicer.util.getNode("ImageLandmarkPoints")
        numFids = ILfidList.GetNumberOfFiducials()
        MLfidList = slicer.util.getNode("ModelLandmarkPoints")

        self.ILData = vtk.vtkPoints()
        for i in xrange(numFids):
            ras = [0, 0, 0]
            pos = ILfidList.GetNthFiducialPosition(i, ras)
            world = [0, 0, 0, 0]
            ILfidList.GetNthFiducialWorldCoordinates(0, world)
            self.ILData.InsertNextPoint(ras[0], ras[1], ras[2])

        self.MLData = vtk.vtkPoints()
        for i in xrange(numFids):
            ras = [0, 0, 0]
            pos = MLfidList.GetNthFiducialPosition(i, ras)
            world = [0, 0, 0, 0]
            MLfidList.GetNthFiducialWorldCoordinates(0, world)
            self.MLData.InsertNextPoint(ras[0], ras[1], ras[2])

        self.ILPointsData = vtk.vtkPolyData()
        self.ILPointsData.SetPoints(self.ILData)

        self.MLPointsData = vtk.vtkPolyData()
        self.MLPointsData.SetPoints(self.MLData)

        source = self.ILPointsData
        target = self.MLPointsData

        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(source)
        icp.SetTarget(target)
        icp.DebugOn()
        icp.SetMaximumNumberOfLandmarks(source.GetNumberOfPoints())
        icp.SetCheckMeanDistance(1)
        icp.SetMaximumMeanDistance(0.00000001)
        icp.GetLandmarkTransform().SetModeToSimilarity()
        icp.StartByMatchingCentroidsOn()
        icp.SetMaximumNumberOfIterations(200)
        icp.GetLandmarkTransform().SetSourceLandmarks(self.ILData)
        icp.GetLandmarkTransform().SetTargetLandmarks(self.MLData)
        icp.Modified()

        transformNode = slicer.vtkMRMLLinearTransformNode()
        transformNode.CanApplyNonLinearTransforms()
        transformNode.SetName('ReferenceImageToOpticalModel')
        matrix = icp.GetLandmarkTransform().GetMatrix()
        transformNode.SetMatrixTransformToParent(matrix)

        slicer.mrmlScene.AddNode(transformNode)
        print "ICP registration Complete."

    def landmarkRegistration(self):
        ILfidList = slicer.util.getNode("ImageLandmarkPoints")
        numFids = ILfidList.GetNumberOfFiducials()
        MLfidList = slicer.util.getNode("ModelLandmarkPoints")

        self.ILData = vtk.vtkPoints()
        for i in xrange(numFids):
            ras = [0, 0, 0]
            pos = ILfidList.GetNthFiducialPosition(i, ras)
            world = [0, 0, 0, 0]
            ILfidList.GetNthFiducialWorldCoordinates(0, world)
            self.ILData.InsertNextPoint(ras[0], ras[1], ras[2])

        self.MLData = vtk.vtkPoints()
        for i in xrange(numFids):
            ras = [0, 0, 0]
            pos = MLfidList.GetNthFiducialPosition(i, ras)
            world = [0, 0, 0, 0]
            MLfidList.GetNthFiducialWorldCoordinates(0, world)
            self.MLData.InsertNextPoint(ras[0], ras[1], ras[2])

        self.ILPointsData = vtk.vtkPolyData()
        self.ILPointsData.SetPoints(self.ILData)

        self.MLPointsData = vtk.vtkPolyData()
        self.MLPointsData.SetPoints(self.MLData)

        source = self.ILData
        target = self.MLData

        lmt = vtk.vtkLandmarkTransform()
        lmt.SetSourceLandmarks(source)
        source.Modified()
        lmt.SetTargetLandmarks(target)
        target.Modified()
        lmt.SetModeToSimilarity()
        lmt.TransformPoints(source, target)

        transformNode = slicer.vtkMRMLLinearTransformNode()
        transformNode.SetName('ReferenceImageToOpticalModel')
        transformNode.CanApplyNonLinearTransforms()
        matrix = lmt.GetMatrix()
        transformNode.SetMatrixTransformToParent(matrix)

        slicer.mrmlScene.AddNode(transformNode)
        print "Landmark Transform completed succesfully."

    # Image Registration functions
    #def callFollowFiducials(self):
     #   fiducialTimer = qt.QTimer()
     #   for delay in xrange(0, 3000, 1000):
      #      fiducialTimer.singleShot(delay, lambda: self.followFiducialCoordinates())

    #def followFiducialCoordinates(self):
     #   ILfidList = slicer.util.getNode('MarkupsFiducial')  # was F
      #  numFids = ILfidList.GetNumberOfFiducials()

       # for i in xrange(numFids):
        #    ras = [0, 0, 0]
         #   pos = ILfidList.GetNthFiducialPosition(i, ras)
        # world = [0, 0, 0, 0]
         #   ILfidList.GetNthFiducialWorldCoordinates(0, world)
          #  self.fiducialMovementDelay = self.fiducialMovementDelay + 1000
           # xcoord = abs(int(ras[0]))
            #ycoord = abs(int(ras[1]))
            #if xcoord < 120 and ycoord < 120:  # maintains that the coordinates stay within the test bed limitations
            #    self.xyMovement(xcoord, ycoord, self.fiducialMovementDelay)
            # ras is the coordinate of the fiducial

    #def findCenterOfMassOfFiducials(self):
     #   ILfidList = slicer.util.getNode('Irregularity')
      #  numFids = ILfidList.GetNumberOfFiducials()
      #  centerOfMass = [0, 0, 0]
      #  sumPos = np.zeros(3)
      #  for i in xrange(numFids):
      #      pos = np.zeros(3)
       #     ILfidList.GetNthFiducialPosition(i, pos)
        #    sumPos += pos
        #centerOfMass = sumPos / numFids
        #xcoord = centerOfMass[0]
        #ycoord = centerOfMass[1]
        #print xcoord, ycoord
        #self.controlledXYMovement(xcoord, ycoord)



                                        # General Printer Movement Commands

    def emergencyStop(self):
        # Writes to the printer to automatically stop all motors
        # Requires reboot
        slicer.modules.openigtlinkremote.logic().SendCommand(self.emergStopCmd, self.serialIGTLNode.GetID())
        self.emergStopCmd.AddObserver(self.emergStopCmd.CommandCompletedEvent, self.onPrinterCommandCompleted)


    def yMovement(self, mvmtDelay, yResolution):
        self.scanTimer = qt.QTimer()
        self.scanTimer.singleShot(mvmtDelay, lambda: self.controlledYMovement(yResolution))

    def XMovement(self, timevar, movevar):
        self.scanTimer = qt.QTimer()
        self.scanTimer.singleShot(timevar, lambda: self.controlledXMovement(movevar))

    def slowEdgeTracing(self, xcoordinate, ycoordinate, timevar):
        self.edgetimer = qt.QTimer()
        self.edgetimer.singleShot(timevar, lambda: self.controlledXYMovement(xcoordinate, ycoordinate))

    def xyMovement(self, xcoordinate, ycoordinate, timevar):
        self.scanTimer = qt.QTimer()
        self.scanTimer.singleShot(timevar, lambda: self.controlledXYMovement(xcoordinate, ycoordinate))

    def ZMovement(self, mvmtDelay, zcoordinate):
        self.scanTimer = qt.QTimer()
        self.scanTimer.singleShot(mvmtDelay, lambda: self.controlledZMovement(zcoordinate))

    def controlledXYMovement(self, xcoordinate, ycoordinate):
        self.xyControlCmd.SetCommandAttribute('Text', 'G1 X%d Y%d' % (xcoordinate, ycoordinate))
        slicer.modules.openigtlinkremote.logic().SendCommand(self.xyControlCmd, self.serialIGTLNode.GetID())

    def controlledXMovement(self, xCoordinate):  # x movement
        self.xControlCmd.SetCommandAttribute('Text', 'G1 X%d' % (xCoordinate))
        slicer.modules.openigtlinkremote.logic().SendCommand(self.xControlCmd, self.serialIGTLNode.GetID())

    def controlledYMovement(self, yCoordinate):  # y movement
        self.yControlCmd.SetCommandAttribute('Text', 'G1 Y%d' % (yCoordinate))
        slicer.modules.openigtlinkremote.logic().SendCommand(self.yControlCmd, self.serialIGTLNode.GetID())

    def controlledZMovement(self, zcoordinate):
        self.zControlCmd.SetCommandAttribute('Text', 'G1 Z%d' % (zcoordinate))
        slicer.modules.openigtlinkremote.logic().SendCommand(self.zControlCmd, self.serialIGTLNode.GetID())

    # specific movement commands for keyboard control, necessary because of serialIGTLNode declaration
    def keyboardControlledXMovementForward(self, serialIGTLNode):  # x movement
        if self.currentXcoordinate < 120:
            self.currentXcoordinate = self.currentXcoordinate + 1
        else:
            self.currentXcoordinate = self.currentXcoordinate - 1
        self.xControlCmd = slicer.vtkSlicerOpenIGTLinkCommand()
        self.xControlCmd.SetCommandName('SendText')
        self.xControlCmd.SetCommandAttribute('DeviceId', "SerialDevice")
        self.xControlCmd.SetCommandTimeoutSec(1.0)
        self.xControlCmd.SetCommandAttribute('Text', 'G1 X%d' % (self.currentXcoordinate))
        slicer.modules.openigtlinkremote.logic().SendCommand(self.xControlCmd, serialIGTLNode.GetID())

    def keyboardControlledXMovementBackwards(self, serialIGTLNode):  # x movement
        if self.currentXcoordinate > 1:
            self.currentXcoordinate = self.currentXcoordinate - 1
        else:
            self.currentXcoordinate = self.currentXcoordinate + 1
        self.xControlCmd = slicer.vtkSlicerOpenIGTLinkCommand()
        self.xControlCmd.SetCommandName('SendText')
        self.xControlCmd.SetCommandAttribute('DeviceId', "SerialDevice")
        self.xControlCmd.SetCommandTimeoutSec(1.0)
        self.xControlCmd.SetCommandAttribute('Text', 'G1 X%d' % (self.currentXcoordinate))
        slicer.modules.openigtlinkremote.logic().SendCommand(self.xControlCmd, serialIGTLNode.GetID())

    def keyboardControlledYMovementForward(self, serialIGTLNode):  # y movement
        if self.currentYcoordinate < 120:
            self.currentYcoordinate = self.currentYcoordinate + 1
        else:
            self.currentYcoordinate = self.currentYcoordinate - 1
        self.yControlCmd = slicer.vtkSlicerOpenIGTLinkCommand()
        self.yControlCmd.SetCommandName('SendText')
        self.yControlCmd.SetCommandAttribute('DeviceId', "SerialDevice")
        self.yControlCmd.SetCommandTimeoutSec(1.0)
        self.yControlCmd.SetCommandAttribute('Text', 'G1 Y%d' % (self.currentYcoordinate))
        slicer.modules.openigtlinkremote.logic().SendCommand(self.yControlCmd, serialIGTLNode.GetID())

    def keyboardControlledYMovementBackwards(self, serialIGTLNode):  # y movement
        if self.currentYcoordinate > 1:
            self.currentYcoordinate = self.currentYcoordinate - 1
        else:
            self.currentYcoordinate = self.currentYcoordinate + 1
        self.yControlCmd = slicer.vtkSlicerOpenIGTLinkCommand()
        self.yControlCmd.SetCommandName('SendText')
        self.yControlCmd.SetCommandAttribute('DeviceId', "SerialDevice")
        self.yControlCmd.SetCommandTimeoutSec(1.0)
        self.yControlCmd.SetCommandAttribute('Text', 'G1 Y%d' % (self.currentYcoordinate))
        slicer.modules.openigtlinkremote.logic().SendCommand(self.yControlCmd, serialIGTLNode.GetID())

    def keyboardControlledHomeMovement(self, serialIGTLNode):
        self.yControlCmd = slicer.vtkSlicerOpenIGTLinkCommand()
        self.yControlCmd.SetCommandName('SendText')
        self.yControlCmd.SetCommandAttribute('DeviceId', "SerialDevice")
        self.yControlCmd.SetCommandTimeoutSec(1.0)
        self.yControlCmd.SetCommandAttribute('Text', 'G28 X Y')
        slicer.modules.openigtlinkremote.logic().SendCommand(self.yControlCmd, serialIGTLNode.GetID())


class PrinterInteractorTest(ScriptedLoadableModuleTest):
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
        self.test_PrinterInteractor1()

    def test_PrinterInteractor1(self):
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
        import urllib
        downloads = (
            ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
        )

        for url, name, loader in downloads:
            filePath = slicer.app.temporaryPath + '/' + name
            if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
                logging.info('Requesting download %s from %s...\n' % (name, url))
                urllib.urlretrieve(url, filePath)
            if loader:
                logging.info('Loading %s...' % (name,))
                loader(filePath)
        self.delayDisplay('Finished with download and loading')

        volumeNode = slicer.util.getNode(pattern="FA")
        logic = PrinterInteractorLogic()
        self.assertIsNotNone(logic.hasImageData(volumeNode))
        self.delayDisplay('Test passed!')