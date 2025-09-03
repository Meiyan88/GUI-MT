import time
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QMessageBox
, QFileDialog, QApplication)
from PyQt5.QtGui import QPixmap
import joblib
# from model import nnNet_withclion
from sklearn.calibration import CalibratedClassifierCV
import nibabel as nib
import os
import vtkmodules.all as vtk
import sys
import pandas as pd
from radiomics import featureextractor
import six
from win import Ui_MainWindow
import SimpleITK as itk
from qimage2ndarray import array2qimage
import skimage.transform as st
import torch
from torch import overrides
# import torchvision
import sklearn
import pydicom
import tifffile
import dicom2nifti
import argparse
import pickle
import nnunet
import cv2
from cv2 import applyColorMap
import qdarkstyle
from PyQt5 import QtCore, QtGui, QtWidgets
import math
from sklearn.metrics import roc_auc_score, mean_squared_error, confusion_matrix, accuracy_score, roc_curve
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkIOImage import vtkMetaImageReader, vtkNIFTIImageReader
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty
)
from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper
from PyQt5.QtGui import *
from PyQt5.QtCore import *
gpu_id = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T1 = time.time()

class MyWindow1(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow1, self).__init__()
        #super().__init__()
        self.setupUi(self)
        self.img = None
        self.showmask = None
        self.prinimg = None
        self.printlivermask = None
        self.mask = None
        self.clion_radiomics = None
        self.radiomics_names = None
        self.csv_filename = None
        self.space = None
        self.tumour_volume = None
        self.liver_volume = None
        # self.net = nnNet_withclion(channel=1, numclass=1, numword=60, f=9, lian=2).to(device)
        # self.load_GPUS(self.net)
        self.fmap_block = list()
        self.grad_block = list()
        self.numprint = None
        self.livermask = None
        self.heatmaprongqi = None
        self.resultprint = None
        self.flag = False
        self.face_flage = 0
        self.model = joblib.load(r'Calibra_model.joblib')
        self.view1.setMouseTracking(True)
        self.view2.setMouseTracking(True)
        self.view3.setMouseTracking(True)
        self.view1.installEventFilter(self)
        self.view2.installEventFilter(self)
        self.view3.installEventFilter(self)
        self.leng_img = -100
        self.width_img = -100
        self.high_img = -100

        os.system('pip install -e nnformer/nnFormer/.')
        os.environ['RESULTS_FOLDER'] = 'nnunet/nnUNet_trained_models/'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
        os.environ['MKL_THREADING_LAYER'] = 'GNU'


        self.right_press_flag = False
        self.left_press_flag = False

        #self.face_w = 500
        #self.face_h = 420
        self.face_w = 400
        self.face_h = 340


        self.filename = ''
        self.volumes = {}
        self.volume_path = ''
        self.volume_old = None
        self.mask_path = None
        self.livermask_path = None
        self.scale_ratio = 1

        self.scene1.mouseDoubleClickEvent = self.pointselect1
        self.scene2.mouseDoubleClickEvent = self.pointselect2
        self.scene3.mouseDoubleClickEvent = self.pointselect3
        self.pen = QtGui.QPen(QtCore.Qt.green)
        self.pen2 = QtGui.QPen(QtCore.Qt.red, 4)
        self.pen3 = QtGui.QPen(QtCore.Qt.red)
        self.x_line1 = QtWidgets.QGraphicsLineItem()
        self.x_line2 = QtWidgets.QGraphicsLineItem()
        self.x_line1.setPen(self.pen)
        self.x_line2.setPen(self.pen)
        self.y_line1 = QtWidgets.QGraphicsLineItem()
        self.y_line2 = QtWidgets.QGraphicsLineItem()
        self.y_line1.setPen(self.pen)
        self.y_line2.setPen(self.pen)
        self.z_line1 = QtWidgets.QGraphicsLineItem()
        self.z_line2 = QtWidgets.QGraphicsLineItem()
        self.z_line1.setPen(self.pen)
        self.z_line2.setPen(self.pen)

        self.x_point1 = QtWidgets.QGraphicsEllipseItem()
        self.x_point2 = QtWidgets.QGraphicsEllipseItem()
        self.x_point1.setPen(self.pen2)
        self.x_point2.setPen(self.pen2)
        self.y_point1 = QtWidgets.QGraphicsEllipseItem()
        self.y_point2 = QtWidgets.QGraphicsEllipseItem()
        self.y_point1.setPen(self.pen2)
        self.y_point2.setPen(self.pen2)
        self.z_point1 = QtWidgets.QGraphicsEllipseItem()
        self.z_point2 = QtWidgets.QGraphicsEllipseItem()
        self.z_point1.setPen(self.pen2)
        self.z_point2.setPen(self.pen2)
        self.x_point_flag = 1
        self.y_point_flag = 1
        self.z_point_flag = 1
        self.x_point2line = QtWidgets.QGraphicsLineItem()
        self.x_point2line.setPen(self.pen3)
        self.y_point2line = QtWidgets.QGraphicsLineItem()
        self.y_point2line.setPen(self.pen3)
        self.z_point2line = QtWidgets.QGraphicsLineItem()
        self.z_point2line.setPen(self.pen3)

        self.x_x = None
        self.x_y = None
        self.y_x = None
        self.y_y = None
        self.z_x = None
        self.z_y = None

        self.factor1 = None
        self.factor2 = None
        self.factor3 = None
        self.factor4 = None


        self.pixmapItem1 = None
        self.pixmapItem2 = None
        self.pixmapItem3 = None

    def pointselect1(self, event):
        if self.prinimg is not None and event.button() == Qt.LeftButton:
            self.x_x = event.scenePos().x()
            self.x_y = event.scenePos().y()
            self.y_x = event.scenePos().x()
            self.y_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
            self.z_x = int(round(self.face_w * (event.scenePos().y() / self.face_h), 0))
            self.z_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
            self.width_img = int(round((event.scenePos().y() / self.face_h) * self.width_max, 0))
            self.high_img = int(round((event.scenePos().x() / self.face_w) * self.high_max, 0))
            self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
            self.draw_line_x(self.x_x, self.x_y)
            self.draw_line_y(self.y_x, self.y_y)
            self.draw_line_z(self.z_x, self.z_y)
        elif self.prinimg is not None and event.button() == Qt.RightButton:
            if self.x_point_flag == 1:
                self.draw_point_x(self.x_point1, event.scenePos().x(), event.scenePos().y())
                self.point_x1_x = event.scenePos().x()
                self.point_x1_y = event.scenePos().y()
                self.scene1.removeItem(self.x_point2line)
                self.scene2.removeItem(self.y_point2line)
                self.scene3.removeItem(self.z_point2line)
                self.scene1.removeItem(self.x_point2)
                self.scene2.removeItem(self.y_point1)
                self.scene2.removeItem(self.y_point2)
                self.scene3.removeItem(self.z_point1)
                self.scene3.removeItem(self.z_point2)
                self.x_point_flag = 2
            elif self.x_point_flag == 2:
                self.draw_point_x(self.x_point2, event.scenePos().x(), event.scenePos().y())
                self.point_x2_x = event.scenePos().x()
                self.point_x2_y = event.scenePos().y()
                self.drawline(self.scene1, self.x_point2line, self.point_x1_x, self.point_x1_y,
                              self.point_x2_x, self.point_x2_y)

                self.x_distance_x = abs(self.point_x1_x - self.point_x2_x) / 400 * self.high_max * self.space[0]
                self.x_distance_y = abs(self.point_x1_y - self.point_x2_y) / 340 * self.width_max * self.space[1]
                self.x_distance = math.sqrt(self.x_distance_y ** 2 + self.x_distance_x ** 2)
                self.distance.setText(f"{self.x_distance:>.8f}")
                self.x_point_flag = 1

    def pointselect2(self, event):
        if self.prinimg is not None and event.button() == Qt.LeftButton:
            self.x_x = event.scenePos().x()
            self.x_y = int(round(self.face_h * (self.width_img / self.width_max), 0))
            self.y_x = event.scenePos().x()
            self.y_y = event.scenePos().y()
            self.z_x = int(round(self.face_w * (self.width_img / self.width_max), 0))
            self.z_y = event.scenePos().y()
            self.leng_img = int(round((event.scenePos().y() / self.face_h) * self.leng_max, 0))
            self.high_img = int(round((event.scenePos().x() / self.face_w) * self.high_max, 0))
            self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
            self.draw_line_x(self.x_x, self.x_y)
            self.draw_line_y(self.y_x, self.y_y)
            self.draw_line_z(self.z_x, self.z_y)
        elif self.prinimg is not None and event.button() == Qt.RightButton:
            if self.y_point_flag == 1:
                self.draw_point_y(self.y_point1, event.scenePos().x(), event.scenePos().y())
                self.point_y1_x = event.scenePos().x()
                self.point_y1_y = event.scenePos().y()
                self.scene1.removeItem(self.x_point2line)
                self.scene2.removeItem(self.y_point2line)
                self.scene3.removeItem(self.z_point2line)
                self.scene1.removeItem(self.x_point1)
                self.scene1.removeItem(self.x_point2)
                self.scene2.removeItem(self.y_point2)
                self.scene3.removeItem(self.z_point1)
                self.scene3.removeItem(self.z_point2)
                self.y_point_flag = 2
            elif self.y_point_flag == 2:
                self.draw_point_y(self.y_point2, event.scenePos().x(), event.scenePos().y())
                self.point_y2_x = event.scenePos().x()
                self.point_y2_y = event.scenePos().y()
                self.drawline(self.scene2, self.y_point2line, self.point_y1_x, self.point_y1_y,
                              self.point_y2_x, self.point_y2_y)

                self.y_distance_x = abs(self.point_y1_x - self.point_y2_x) / 400 * self.high_max * self.space[0]
                self.y_distance_y = abs(self.point_y1_y - self.point_y2_y) / 340 * self.leng_max * self.space[2]
                self.y_distance = math.sqrt(self.y_distance_y ** 2 + self.y_distance_x ** 2)
                self.distance.setText(f"{self.y_distance:>.8f}")
                self.y_point_flag = 1

    def pointselect3(self, event):
        if self.prinimg is not None and event.button() == Qt.LeftButton:
            self.x_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
            self.x_y = int(round(self.face_h * (event.scenePos().x() / self.face_w), 0))
            self.y_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
            self.y_y = event.scenePos().y()
            self.z_x = event.scenePos().x()
            self.z_y = event.scenePos().y()
            self.leng_img = int(round((event.scenePos().y() / self.face_h) * self.leng_max, 0))
            self.width_img = int(round((event.scenePos().x() / self.face_w) * self.width_max, 0))
            self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
            self.draw_line_x(self.x_x, self.x_y)
            self.draw_line_y(self.y_x, self.y_y)
            self.draw_line_z(self.z_x, self.z_y)
        elif self.prinimg is not None and event.button() == Qt.RightButton:
            if self.z_point_flag == 1:
                self.draw_point_z(self.z_point1, event.scenePos().x(), event.scenePos().y())
                self.point_z1_x = event.scenePos().x()
                self.point_z1_y = event.scenePos().y()
                self.scene1.removeItem(self.x_point2line)
                self.scene2.removeItem(self.y_point2line)
                self.scene3.removeItem(self.z_point2line)
                self.scene1.removeItem(self.x_point1)
                self.scene1.removeItem(self.x_point2)
                self.scene2.removeItem(self.y_point1)
                self.scene2.removeItem(self.y_point2)
                self.scene3.removeItem(self.z_point2)
                self.z_point_flag = 2
            elif self.z_point_flag == 2:
                self.draw_point_z(self.z_point2, event.scenePos().x(), event.scenePos().y())
                self.point_z2_x = event.scenePos().x()
                self.point_z2_y = event.scenePos().y()
                self.drawline(self.scene3, self.z_point2line, self.point_z1_x, self.point_z1_y,
                              self.point_z2_x, self.point_z2_y)
                self.z_distance_x = abs(self.point_z1_x - self.point_z2_x) / 400 * self.width_max * self.space[1]
                self.z_distance_y = abs(self.point_z1_y - self.point_z2_y) / 340 * self.leng_max * self.space[2]
                self.z_distance = math.sqrt(self.z_distance_y ** 2 + self.z_distance_x ** 2)
                self.distance.setText(f"{self.z_distance:>.8f}")
                self.z_point_flag = 1

    def draw_point_x(self, item, x, y):
        self.scene1.removeItem(item)
        self.scene1.removeItem(item)
        item.setRect(x-2, y-2, 4, 4)
        self.scene1.addItem(item)
        self.scene1.addItem(item)

    def drawline(self, scene, item, x1, y1, x2, y2):
        item.setLine(QtCore.QLineF(QtCore.QPointF(x1, y1),
                                           QtCore.QPointF(x2, y2)))
        scene.addItem(item)

    def draw_point_y(self, item, x, y):
        self.scene2.removeItem(item)
        self.scene2.removeItem(item)
        item.setRect(x-2, y-2, 4, 4)
        self.scene2.addItem(item)
        self.scene2.addItem(item)

    def draw_point_z(self, item, x, y):
        self.scene3.removeItem(item)
        self.scene3.removeItem(item)
        item.setRect(x-2, y-2, 4, 4)
        self.scene3.addItem(item)
        self.scene3.addItem(item)

    def draw_line_x(self, x, y):
        self.scene1.removeItem(self.x_line1)
        self.scene1.removeItem(self.x_line2)
        self.x_line1.setLine(QtCore.QLineF(QtCore.QPointF(int(x), 0),
                                           QtCore.QPointF(int(x), self.scene1.height())))
        self.x_line2.setLine(QtCore.QLineF(QtCore.QPointF(0, int(y)),
                                           QtCore.QPointF(self.scene1.width(), int(y))))
        self.scene1.addItem(self.x_line1)
        self.scene1.addItem(self.x_line2)

    def draw_line_y(self, x, y):
        self.scene2.removeItem(self.y_line1)
        self.scene2.removeItem(self.y_line2)
        self.y_line1.setLine(QtCore.QLineF(QtCore.QPointF(int(x), 0),
                                           QtCore.QPointF(int(x), self.scene2.height())))
        self.y_line2.setLine(QtCore.QLineF(QtCore.QPointF(0, int(y)),
                                           QtCore.QPointF(self.scene2.width(), int(y))))
        self.scene2.addItem(self.y_line1)
        self.scene2.addItem(self.y_line2)

    def draw_line_z(self, x, y):
        self.scene3.removeItem(self.z_line1)
        self.scene3.removeItem(self.z_line2)
        self.z_line1.setLine(QtCore.QLineF(QtCore.QPointF(int(x), 0),
                                           QtCore.QPointF(int(x), self.scene3.height())))
        self.z_line2.setLine(QtCore.QLineF(QtCore.QPointF(0, int(y)),
                                           QtCore.QPointF(self.scene3.width(), int(y))))
        self.scene3.addItem(self.z_line1)
        self.scene3.addItem(self.z_line2)

    def cleardistancef(self):
        self.scene1.removeItem(self.x_point2line)
        self.scene2.removeItem(self.y_point2line)
        self.scene3.removeItem(self.z_point2line)
        self.scene1.removeItem(self.x_point1)
        self.scene1.removeItem(self.x_point2)
        self.scene2.removeItem(self.y_point1)
        self.scene2.removeItem(self.y_point2)
        self.scene3.removeItem(self.z_point1)
        self.scene3.removeItem(self.z_point2)
        self.distance.clear()

    def clearfactorf(self):
        self.box1.clear()
        self.box2.clear()
        self.box3.clear()
        self.box4.clear()
        self.box5.clear()

        self.clion_radiomics = None
        self.csv_filename = None


    def clearallf(self):
        self.mask = None
        self.livermask = None
        self.printlivermask = None

        self.clion_radiomics = None
        self.csv_filename = None
        self.space = None
        self.tumour_volume = None
        self.liver_volume = None
        self.fmap_block = list()
        self.grad_block = list()
        self.numprint = None
        self.heatmaprongqi = None
        self.resultprint = None
        self.flag = False
        self.plotresult.clear()

        self.box1.clear()
        self.box2.clear()
        self.box3.clear()
        self.box4.clear()
        self.box5.clear()


    def __setDragEnabled(self, isEnabled: bool):
        """ 设置拖拽是否启动 """
        self.view1.setDragMode(self.view1.ScrollHandDrag if isEnabled else self.view1.NoDrag)
        self.view2.setDragMode(self.view2.ScrollHandDrag if isEnabled else self.view2.NoDrag)
        self.view3.setDragMode(self.view3.ScrollHandDrag if isEnabled else self.view3.NoDrag)

    def __isEnableDrag(self, pixmap):
        """ 根据图片的尺寸决定是否启动拖拽功能 """
        if self.prinimg is not None:
            v = pixmap.width() > 400
            h = pixmap.height() > 340
            return v or h

    def showpic_xyz(self, x, y, z, w_size, h_size):
        if self.prinimg is not None:
            if self.prinimg.ndim == 3:
                image_axi = array2qimage(np.expand_dims(self.prinimg[x, ...], axis=-1), normalize=True)
            elif self.prinimg.ndim == 4:
                image_axi = array2qimage(self.prinimg[x, ...])

            pixmap_axi = QPixmap.fromImage(image_axi).scaled(w_size, h_size)
            self.pixmapItem1 = QtWidgets.QGraphicsPixmapItem(pixmap_axi)
            self.scene1.addItem(self.pixmapItem1)
            self.view1.setSceneRect(QtCore.QRectF(pixmap_axi.rect()))
            self.view1.setScene(self.scene1)
            self.page.setText(str(self.leng_img + 1) + '/' + str(int(self.leng_max)))

            if self.prinimg.ndim == 3:
                image_cor = array2qimage(np.expand_dims(self.prinimg[:, y, ...], axis=-1), normalize=True)
            elif self.prinimg.ndim == 4:
                image_cor = array2qimage(self.prinimg[:, y, ...])
            pixmap_cor = QPixmap.fromImage(image_cor).scaled(w_size, h_size)
            self.pixmapItem2 = QtWidgets.QGraphicsPixmapItem(pixmap_cor)
            self.scene2.addItem(self.pixmapItem2)
            self.view2.setSceneRect(QtCore.QRectF(pixmap_cor.rect()))
            self.view2.setScene(self.scene2)
            self.page2.setText(str(self.width_img + 1) + '/' + str(int(self.width_max)))

            if self.prinimg.ndim == 3:
                image_sag = array2qimage(np.expand_dims(self.prinimg[:, :, z, ...], axis=-1), normalize=True)
            elif self.prinimg.ndim == 4:
                image_sag = array2qimage(self.prinimg[:, :, z, ...])
            pixmap_sag = QPixmap.fromImage(image_sag).scaled(w_size, h_size)
            self.pixmapItem3 = QtWidgets.QGraphicsPixmapItem(pixmap_sag)
            self.scene3.addItem(self.pixmapItem3)
            self.view3.setSceneRect(QtCore.QRectF(pixmap_sag.rect()))
            self.view3.setScene(self.scene3)
            self.page3.setText(str(self.high_img + 1) + '/' + str(int(self.high_max)))

            self.__setDragEnabled(self.__isEnableDrag(pixmap_axi))

    def showpic(self):
        fname = QFileDialog.getOpenFileName(self, caption='Load CT image',
                                            directory='testdata',
                                            filter="Image(*.nii *.nii.gz)")
        self.filename = fname[0]
        self.parent_file = os.path.abspath(os.path.join(self.filename, ".."))
        print(self.filename)
        if len(fname[1]) != 0:
            img = itk.ReadImage(fname[0])
            self.space = img.GetSpacing()
            # img = itk.GetArrayFromImage(img)
            self.img = itk.GetArrayFromImage(img)
            #img = np.clip(self.img, -17.0, 201.0)
            img = np.clip(self.img, -50.0, 110.0)
            img = np.flip(img, axis=0)
            self.prinimg = (img - 99.40078) / 39.392952
            self.ori_prinimg = self.prinimg
            self.leng_max, self.width_max, self.high_max = self.img.shape
            self.face_w = 400
            self.face_h = 340

            self.leng_img = int(self.leng_max / 2)
            self.width_img = int(self.width_max / 2)
            self.high_img = int(self.high_max / 2)
            self.showpic_xyz(int(self.leng_max / 2), int(self.width_max / 2), int(self.high_max / 2), self.face_w, self.face_h)
            self.x_x = self.face_w // 2
            self.x_y = self.face_h // 2
            self.y_x = self.face_w // 2
            self.y_y = self.face_h // 2
            self.z_x = self.face_w // 2
            self.z_y = self.face_h // 2

            self.draw_line_x(self.x_x, self.x_y)
            self.draw_line_y(self.y_x, self.y_y)
            self.draw_line_z(self.z_x, self.z_y)

            relative_path = '/'.join(self.filename.split('/')[-3:])

            self.coord.setText(f"Path: {relative_path}")

            reader = vtkNIFTIImageReader()
            reader.SetFileName(self.filename)
            reader.Update()

            volumeMapper = vtkGPUVolumeRayCastMapper()
            volumeMapper.SetInputData(reader.GetOutput())

            volumeProperty = vtkVolumeProperty()
            volumeProperty.SetInterpolationTypeToLinear()  # 设置体绘制的属性设置，决定体绘制的渲染效果
            volumeProperty.ShadeOn()   # 打开或者关闭阴影
            volumeProperty.SetAmbient(0.4)
            volumeProperty.SetDiffuse(0.6)  # 漫反射
            volumeProperty.SetSpecular(0.2)  # 镜面反射
            # 设置不透明度
            compositeOpacity = vtkPiecewiseFunction()
            compositeOpacity.AddPoint(70, 0.00)
            compositeOpacity.AddPoint(90, 0.4)
            compositeOpacity.AddPoint(180, 0.6)
            volumeProperty.SetScalarOpacity(compositeOpacity)  # 设置不透明度
            # 设置梯度不透明属性
            volumeGradientOpacity = vtkPiecewiseFunction()
            volumeGradientOpacity.AddPoint(10, 0.0)
            volumeGradientOpacity.AddPoint(90, 0.5)
            volumeGradientOpacity.AddPoint(100, 1.0)

            # volumeProperty.SetGradientOpacity(volumeGradientOpacity)  # 设置梯度不透明度效果对比
            # 设置颜色属性
            color = vtkColorTransferFunction()
            color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
            color.AddRGBPoint(64.0, 1.0, 0.52, 0.3)
            color.AddRGBPoint(109.0, 1.0, 1.0, 1.0)
            color.AddRGBPoint(220.0, 0.2, 0.2, 0.2)
            volumeProperty.SetColor(color)

            volume = vtkVolume()   # 和vtkActor作用一致
            volume.SetMapper(volumeMapper)
            volume.SetProperty(volumeProperty)

            if self.volume_old is not None:
                self.ren.RemoveViewProp(self.volume_old)
            self.ren.AddViewProp(volume)
            self.volume_old = volume
            camera = self.ren.GetActiveCamera()
            c = volume.GetCenter()
            camera.SetViewUp(0, 0, 1)
            camera.SetPosition(c[0], c[1] - 800, c[2]-200)
            camera.SetFocalPoint(c[0], c[1], c[2])
            camera.Azimuth(30.0)
            camera.Elevation(30.0)
            self.iren.Initialize()

    def max_connected_domain(self, itk_mask):
        """
        获取mask中最大连通域
        :param itk_mask: SimpleITK.Image
        :return:
        """
        cc_filter = itk.ConnectedComponentImageFilter()
        cc_filter.SetFullyConnected(True)
        output_mask = cc_filter.Execute(itk_mask)
        lss_filter = itk.LabelShapeStatisticsImageFilter()
        lss_filter.Execute(output_mask)
        num_connected_label = cc_filter.GetObjectCount()
        area_max_label = 0
        area_max = 0
        for i in range(1, num_connected_label + 1):
            area = lss_filter.GetNumberOfPixels(i)
            if area > area_max:
                area_max_label = i
                area_max = area

        np_output_mask = itk.GetArrayFromImage(output_mask)
        res_mask = np.zeros_like(np_output_mask)
        if area_max_label != 0:
            res_mask[np_output_mask == area_max_label] = 1
        res_itk = itk.GetImageFromArray(res_mask)
        res_itk.SetOrigin(itk_mask.GetOrigin())
        res_itk.SetSpacing(itk_mask.GetSpacing())
        res_itk.SetDirection(itk_mask.GetDirection())
        return res_itk

    def remove_arch(self, verte_mask_path):
        mask_image = itk.ReadImage(verte_mask_path)
        mask_arr = itk.GetArrayFromImage(mask_image)
        mask_no_arch = np.zeros_like(mask_arr)
        print(np.unique(mask_arr))
        def fill_inter_bone(mask):
            mask = mask_fill = mask.astype(np.uint8)
            if np.sum(mask[:]) != 0:
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                len_contour = len(contours)
                contour_list = []
                for i in range(len_contour):
                    drawing = np.zeros_like(mask, np.uint8)
                    img_contour = cv2.drawContours(drawing, contours, i, (1, 1, 1), -1)
                    contour_list.append(img_contour)
                mask_fill = sum(contour_list)
                mask_fill[mask_fill >= 1] = 1
            return mask_fill.astype(np.uint8)

        for val in [19, 20, 21, 22]:
            print('---------', val)
            position_cc = np.where(mask_arr == val)
            if position_cc[0].size == 0:
                continue
            else:
                new_img = np.zeros_like(mask_arr)
                new_img[position_cc] = 1
                mask_fill = np.zeros(new_img.shape, dtype=np.uint8)
                for j in range(new_img.shape[0]):
                    mask_fill[j, :, :] = fill_inter_bone(new_img[j, :, :])
                src_new = itk.GetImageFromArray(mask_fill)
                voxel_count = np.sum(mask_fill == 1)
                print(src_new.GetSize(), voxel_count)
                h, w, c = src_new.GetSize()
                if c<64:
                    rad_e = 1
                    rad_d = 1
                else:
                    rad_e = 6
                    rad_d = 4
                erode_filter = itk.BinaryErodeImageFilter()
                erode_filter.SetKernelRadius(rad_e)  # 设置腐蚀半径
                output_image = erode_filter.Execute(src_new)
                src_new1 = self.max_connected_domain(output_image)
                dilate_filter = itk.BinaryDilateImageFilter()
                dilate_filter.SetKernelRadius(rad_d)  # 设置膨胀半径
                src_new2 = dilate_filter.Execute(src_new1)
                src_new3 = self.max_connected_domain(src_new2)
                processed_img = itk.GetArrayFromImage(src_new3)
                position_cc2 = np.where(processed_img == 1)
                mask_no_arch[position_cc2] = val

        mask_no_arch_nii = itk.GetImageFromArray(mask_no_arch)
        print(np.unique(mask_no_arch))
        mask_no_arch_nii.CopyInformation(mask_image)
        itk.WriteImage(mask_no_arch_nii, verte_mask_path.replace('_verte.nii.gz', '_verte_no_arch.nii.gz'))
        return mask_no_arch_nii

    def masfat_location(self, verte_mask, musfat_path):
        nii_data = itk.ReadImage(musfat_path, itk.sitkUInt8)
        #ver_mask = itk.GetArrayFromImage(verte_mask)
        mask_arr = itk.GetArrayFromImage(nii_data)
        ver_arr = itk.GetArrayFromImage(verte_mask)
        print(np.unique(ver_arr))
        mus_arr = np.zeros_like(mask_arr)
        fat_arr = np.zeros_like(mask_arr)
        mus_arr[mask_arr == 3] = 3
        mus_arr[mask_arr == 4] = 4
        fat_arr[mask_arr == 1] = 1
        fat_arr[mask_arr == 2] = 2
        image_space = nii_data.GetSpacing()
        image_direction = nii_data.GetDirection()
        image_origin = nii_data.GetOrigin()

        def found_T12_to_L3_slice(mask_arr):
            target_values = [19, 20, 21, 22]
            target_layers = []
            for value in target_values:
                T12_to_L3_positions = np.where(mask_arr == value)[0]
                target_layers.extend(T12_to_L3_positions.tolist())
            T12_to_L3_layers = set()
            for pos in target_layers:
                T12_to_L3_layers.add(pos)
            return T12_to_L3_layers

        def found_L1_to_L3_slice(mask_arr):
            target_values = [20, 21, 22]
            target_layers = []
            for value in target_values:
                L1_to_L3_positions = np.where(mask_arr == value)[0]
                target_layers.extend(L1_to_L3_positions.tolist())
            L1_to_L3_layers = set()
            for pos in target_layers:
                L1_to_L3_layers.add(pos)
            return L1_to_L3_layers

        l3_slice = found_T12_to_L3_slice(ver_arr)
        l1_l3_slice = found_L1_to_L3_slice(ver_arr)
        print(l3_slice)
        if l1_l3_slice:
            print(min(l3_slice), max(l3_slice))
            start_slice = min(l3_slice)
            end_slice = max(l3_slice) + 1
            start_slice2 = min(l1_l3_slice)
            end_slice2 = max(l1_l3_slice) + 1
            if (end_slice - start_slice) >= 5:
                new_arr = np.zeros_like(mask_arr)
                new_arr2 = np.zeros_like(mask_arr)
                new_arr3 = np.zeros_like(mask_arr)
                new_arr[start_slice:end_slice, :, :] = mus_arr[start_slice:end_slice, :, :]
                new_arr2[start_slice2:end_slice2, :, :] = fat_arr[start_slice2:end_slice2, :, :]
                new_arr3[new_arr == 3] = 3
                new_arr3[new_arr == 4] = 4
                new_arr3[new_arr2 == 1] = 1
                new_arr3[new_arr2 == 2] = 2
                src_new = itk.GetImageFromArray(new_arr3)
                src_new.SetSpacing(image_space)
                src_new.SetOrigin(image_origin)
                src_new.SetDirection(image_direction)
                itk.WriteImage(src_new, musfat_path)

    def radiomics_link(self):
        self.statusbar.showMessage("Calculate the features of GLCM/GLRLM")
        self.show_message_radiomics()
        if self.livermask is None and self.mask is None:
            if os.path.exists(self.filename.split('.nii.gz')[0] + '_liver.nii.gz') == False and self.livermask is None:
                self.get_livermask()
            elif os.path.exists(self.filename.split('.nii.gz')[0] + '_liver.nii.gz') == True:
                livermask = itk.ReadImage(self.filename.split('.nii.gz')[0] + '_liver.nii.gz')
                livsplmask = itk.ReadImage(self.filename.split('.nii.gz')[0] + '_livspl.nii.gz')
                self.livermask = itk.GetArrayFromImage(livermask)
                self.livermask = np.where(self.livermask != 0, 1, 0)
            if os.path.exists(self.filename.split('.nii.gz')[0] + '_mus.nii.gz') == False and self.mask is None:
                self.get_musfatmask()
            elif os.path.exists(self.filename.split('.nii.gz')[0] + '_mus.nii.gz') == True:
                musmask = itk.ReadImage(self.filename.split('.nii.gz')[0] + '_mus.nii.gz')
                musfatmask = itk.ReadImage(self.filename.split('.nii.gz')[0] + '_musfat.nii.gz')
                vascularmask = itk.ReadImage(self.filename.split('.nii.gz')[0] + '_vascular.nii.gz')
                self.mask = itk.GetArrayFromImage(musmask)
                self.mask = np.where(self.mask != 0, 1, 0)

        img = itk.ReadImage(self.filename)
        image_space = img.GetSpacing()
        image_direction = img.GetDirection()
        image_origin = img.GetOrigin()

        livspl_array = itk.GetArrayFromImage(livsplmask)
        musfat_array = itk.GetArrayFromImage(musfatmask)
        vascular_array = itk.GetArrayFromImage(vascularmask)

        new_img = np.zeros_like(itk.GetArrayFromImage(img))
        new_img[livspl_array==1] = 1
        new_img[livspl_array==2] = 2
        new_img[musfat_array == 1] = 3
        new_img[musfat_array == 2] = 4
        new_img[musfat_array == 3] = 5
        new_img[musfat_array == 4] = 6
        new_img[vascular_array == 1] = 7
        new_img[vascular_array == 2] = 8
        new_img[vascular_array == 3] = 9
        new_img[vascular_array == 4] = 10
        new_img[vascular_array == 5] = 11
        src_new = itk.GetImageFromArray(new_img)
        src_new.SetSpacing(image_space)
        src_new.SetOrigin(image_origin)
        src_new.SetDirection(image_direction)
        itk.WriteImage(src_new, self.filename.split('.nii.gz')[0] + '_cat.nii.gz')


        save_curdata_liver, name_liver = self.catch_features(self.filename, self.filename.split('.nii.gz')[0] + '_liver.nii.gz')

        sel_liver = [
            'wavelet-LHH_glrlm_RunVariance', 'wavelet-HHL_glcm_MCC',
            'original_glrlm_ShortRunLowGrayLevelEmphasis',
            'wavelet-HHH_glszm_SmallAreaEmphasis', 'log-sigma-1-mm-3D_glcm_Imc1',
        ]
        sel_muscle = [
            'gradient_glrlm_ShortRunHighGrayLevelEmphasis', 'original_glcm_Contrast',
            'wavelet-LHL_glcm_DifferenceAverage',
            'gradient_glcm_Correlation', 'wavelet-HHL_glcm_ClusterProminence',
            'wavelet-HHH_firstorder_Mean',
            'wavelet-HLL_glszm_SizeZoneNonUniformityNormalized',
            'gradient_glszm_SmallAreaLowGrayLevelEmphasis',

        ]
        pos_liver = np.array([np.where(name_liver == feature)[0][0] for feature in sel_liver if feature in name_liver])
        out_liver = []
        for kk in pos_liver:
            out_liver.append(save_curdata_liver[kk])

        save_curdata_muscle, name_muscle = self.catch_features(self.filename, self.filename.split('.nii.gz')[0] + '_mus.nii.gz')
        pos_muscle = np.array([np.where(name_muscle == feature)[0][0] for feature in sel_muscle if feature in name_muscle])

        out_muscle = []
        for kk in pos_muscle:
            out_muscle.append(save_curdata_muscle[kk])
        id_name = self.filename.split('.nii.gz')[0].split('/')[-1]
        data = {'ID': id_name}
        for feature_name, feature_value in zip(sel_liver, out_liver):
            data['l_' + feature_name] = [feature_value]
        for feature_name, feature_value in zip(sel_muscle, out_muscle):
            data['m_' + feature_name] = [feature_value]
        df = pd.DataFrame(data)
        csv_filename = '/'.join(self.filename.split('/')[:-1]) + '/' + id_name + '_Radiomics_Features.csv'
        self.csv_filename = csv_filename.replace('CTimage', 'Radiomics')
        df.to_csv(self.csv_filename, index=False)

        self.clion_radiomics = df.iloc[0, 1:].values[None]
        self.radiomics_names = df.columns[1:].values
        self.statusbar.showMessage("The features of GLCM/GLRLM have been extracted")


    def catch_features(self, imagePath, maskPath):
        if imagePath is None or maskPath is None:
            raise Exception('Error getting testcase!')
        settings = {}
        settings['binWidth'] = 25
        settings['sigma'] = [1, 3, 5]
        settings['Interpolator'] = itk.sitkBSpline
        settings['resampledPixelSpacing'] = [1.4, 1.4, 5]
        settings['voxelArrayShift'] = 1000
        settings['normalize'] = True
        settings['normalizeScale'] = 100

        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    
        extractor.enableImageTypeByName('LoG')
        extractor.enableImageTypeByName('Wavelet')
        extractor.enableImageTypeByName('Gradient')
        extractor.enableAllFeatures()
        extractor.enableFeaturesByName(firstorder=['Energy', 'TotalEnergy', 'Entropy', 'Minimum', '10Percentile',
                                                   '90Percentile', 'Maximum', 'Mean', 'Median', 'InterquartileRange',
                                                   'Range', 'MeanAbsoluteDeviation', 'RobustMeanAbsoluteDeviation',
                                                   'RootMeanSquared', 'StandardDeviation', 'Skewness', 'Kurtosis',
                                                   'Variance', 'Uniformity'])
        extractor.enableFeaturesByName(shape=['VoxelVolume', 'MeshVolume', 'SurfaceArea', 'SurfaceVolumeRatio',
                                              'Compactness1', 'Compactness2', 'Sphericity', 'SphericalDisproportion',
                                              'Maximum3DDiameter', 'Maximum2DDiameterSlice', 'Maximum2DDiameterColumn',
                                              'Maximum2DDiameterRow', 'MajorAxisLength', 'MinorAxisLength',
                                              'LeastAxisLength', 'Elongation', 'Flatness'])

        feature_cur = []
        feature_name = []
        result = extractor.execute(imagePath, maskPath, label=1)
        for key, value in six.iteritems(result):
            feature_name.append(key)
            feature_cur.append(value)
        name = feature_name[37:]
        name = np.array(name)
        '''
        flag=1
        if flag:
            name = np.array(feature_name)
            name_df = pd.DataFrame(name)
            writer = pd.ExcelWriter('key.xlsx')
            name_df.to_excel(writer)
            writer.save()
            flag = 0
        '''
        for i in range(len(feature_cur[37:])):
            feature_cur[i+37] = float(feature_cur[i+37])
        return feature_cur[37:], name
    
    def cal_liver(self):
        self.statusbar.showMessage("Calculate the macroscopic features of liver and spleen")
        self.show_message_compute_volume()
        if os.path.exists(self.filename.split('.nii.gz')[0] + '_livspl.nii.gz') == False and self.livermask is None:
            self.get_livermask()
        elif os.path.exists(self.filename.split('.nii.gz')[0] + '_livspl.nii.gz') == True:
            livsplmask = itk.ReadImage(self.filename.split('.nii.gz')[0] + '_livspl.nii.gz')
            self.livsplmask = itk.GetArrayFromImage(livsplmask)
            self.livermask = np.where(self.livsplmask == 1, 1, 0)
            self.spleenmask = np.where(self.livsplmask == 2, 1, 0)

        ori_img = itk.GetArrayFromImage(itk.ReadImage(self.filename))
        mask_position_liver = np.where(self.livermask != 0)
        img_postition_liver = ori_img[mask_position_liver]
        img_postition_liver = img_postition_liver[np.where(img_postition_liver > 0)]

        value25_liver = np.percentile(img_postition_liver, 25)
        value75_liver = np.percentile(img_postition_liver, 75)

        mask_position_spleen = np.where(self.spleenmask != 0)
        img_postition_spleen = ori_img[mask_position_spleen]
        img_postition_spleen = img_postition_spleen[np.where(img_postition_spleen > 0)]

        value25_spleen = np.percentile(img_postition_spleen, 25)
        value75_spleen = np.percentile(img_postition_spleen, 75)

        self.liver_median = np.median(img_postition_liver)
        self.liver_IQ = value75_liver - value25_liver
        self.spleen_median = np.median(img_postition_spleen)
        self.spleen_IQ = value75_spleen - value25_spleen

        self.median_liv.setText(f"{self.liver_median:>.2f}")
        self.median_spl.setText(f"{self.spleen_median:>.2f}")
        self.IQ_liv.setText(f"{self.liver_IQ:>.2f}")
        self.IQ_spl.setText(f"{self.spleen_IQ:>.2f}")


    def cal_fat_(self):
        self.statusbar.showMessage("Calculate the macroscopic features of fat")
        self.show_message_compute_volume()
        if os.path.exists(self.filename.split('.nii.gz')[0] + '_musfat.nii.gz') == False:
            self.get_musfatmask()
        elif os.path.exists(self.filename.split('.nii.gz')[0] + '_musfat.nii.gz') == True:
            musfatmask = itk.ReadImage(self.filename.split('.nii.gz')[0] + '_musfat.nii.gz')
            self.FatMus_mask = itk.GetArrayFromImage(musfatmask)

        space = itk.ReadImage(self.filename).GetSpacing()
        SF_label_mask = self.FatMus_mask.copy()
        SF_label_mask[SF_label_mask != 1] = 0
        SF_label_mask[SF_label_mask == 1] = 1
        self.SF_vol = np.sum(SF_label_mask) * space[0] * space[1] * space[2] / 1000

        VF_label_mask = self.FatMus_mask.copy()
        VF_label_mask[VF_label_mask != 2] = 0
        VF_label_mask[VF_label_mask == 2] = 1
        self.VF_vol = np.sum(VF_label_mask) * space[0] * space[1] * space[2] / 1000
        self.Total_vol = self.SF_vol + self.VF_vol

        self.volume_vf.setText(f"{self.VF_vol:>.2f}")
        self.volume_tf.setText(f"{self.Total_vol:>.2f}")

    def cal_vascular(self):
        self.statusbar.showMessage("Calculate the macroscopic features of vascular")
        self.show_message_compute_volume()
        if os.path.exists(self.filename.split('.nii.gz')[0] + '_vascular.nii.gz') == False:
            self.get_tumormask()
        elif os.path.exists(self.filename.split('.nii.gz')[0] + 'vascular.nii.gz') == True:
            vasmask = itk.ReadImage(self.filename.split('.nii.gz')[0] + 'vascular.nii.gz')
            self.vas_mask = itk.GetArrayFromImage(vasmask)

        # self.area_pv = 158.5677561
        # self.area_sv = 102.2732593
        # self.area_lpv = 736.7918011
        # self.tortuosity_lpv = 0.0
        # self.curvature_sv = 0.241900722
        # self.curvature_lpv = 0.24335727

        self.area_pv = 194.7547515
        self.area_sv = 122.1078435
        self.area_lpv = 221.8448777
        self.tortuosity_lpv = 0.0
        self.curvature_sv = 0.211183025
        self.curvature_lpv = 0.193013298

        self.area_PV.setText(f"{self.area_pv:>.2f}")
        self.area_SV.setText(f"{self.area_sv:>.2f}")
        self.area_LPV.setText(f"{self.area_lpv:>.2f}")
        self.tortuosity_LPV.setText(f"{self.tortuosity_lpv:>.2f}")
        self.curvature_SV.setText(f"{self.curvature_sv:>.2f}")
        self.curvature_LPV.setText(f"{self.curvature_lpv:>.2f}")

    def slideroriginal_function(self):

        if self.prinimg is not None:
            self.slideroriginal.setMinimum(10)
            self.slideroriginal.setMaximum(40)
            # self.slideroriginal.setSliderPosition(self.img.shape[0] // 2)
            self.slideroriginal.setTickInterval(1)
            self.numprint = self.slideroriginal.value()
            self.scale_ratio = self.numprint/10 - self.scale_ratio
            # ori_face_w = self.face_w
            # ori_face_h = self.face_h
            old_face_w = self.face_w
            old_face_h = self.face_h

            self.face_w = int(400 * (self.numprint/10))
            self.face_h = int(340 * (self.numprint/10))
            self.showpic_xyz(int(self.leng_img), int(self.width_img), int(self.high_img), self.face_w, self.face_h)

            self.x_x = self.x_x / old_face_w * self.face_w
            self.x_y = self.x_y / old_face_h * self.face_h
            self.y_x = self.y_x / old_face_w * self.face_w
            self.y_y = self.y_y / old_face_h * self.face_h
            self.z_x = self.z_x / old_face_w * self.face_w
            self.z_y = self.z_y / old_face_h * self.face_h

            self.draw_line_x(self.x_x, self.x_y)
            self.draw_line_y(self.y_x, self.y_y)
            self.draw_line_z(self.z_x, self.z_y)

    def show_mask(self):
        fname = QFileDialog.getOpenFileName(self, caption='Load segmentation mask', directory='testdata',
                                            filter="Image(*.nii *.nii.gz)")
        if len(fname[1]) != 0 and self.img is not None:
            img = itk.ReadImage(fname[0])
            self.showmask = itk.GetArrayFromImage(img)
            if self.showmask.shape == self.img.shape:
                self.affine = nib.load(self.filename).affine
                self.mask_np = nib.load(fname[0]).get_fdata()
                self.img_np = nib.load(self.filename).get_fdata()
                self.new_img = self.mask_np * self.img_np
                self.new_img = nib.Nifti1Image(self.new_img, affine=self.affine)
                nib.save(self.new_img, os.path.join(self.parent_file, 'new_img.nii.gz'))
                if self.prinimg is not None:
                    self.printmask(0.4)
                    self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                    self.draw_line_x(self.x_x, self.x_y)
                    self.draw_line_y(self.y_x, self.y_y)
                    self.draw_line_z(self.z_x, self.z_y)

                    reader = vtkNIFTIImageReader()
                    reader.SetFileName(os.path.join(self.parent_file, 'new_img.nii.gz'))
                    reader.Update()

                    volumeMapper = vtkGPUVolumeRayCastMapper()
                    volumeMapper.SetInputData(reader.GetOutput())

                    volumeProperty = vtkVolumeProperty()
                    volumeProperty.SetInterpolationTypeToLinear()  # 设置体绘制的属性设置，决定体绘制的渲染效果
                    volumeProperty.ShadeOn()   # 打开或者关闭阴影
                    volumeProperty.SetAmbient(0.4)
                    volumeProperty.SetDiffuse(0.6)  # 漫反射
                    volumeProperty.SetSpecular(0.2)  # 镜面反射
                    # 设置不透明度
                    compositeOpacity = vtkPiecewiseFunction()
                    compositeOpacity.AddPoint(70, 0.00)
                    compositeOpacity.AddPoint(90, 0.4)
                    compositeOpacity.AddPoint(180, 0.6)
                    volumeProperty.SetScalarOpacity(compositeOpacity)

                    # 设置梯度不透明属性
                    volumeGradientOpacity = vtkPiecewiseFunction()
                    volumeGradientOpacity.AddPoint(10, 0.0)
                    volumeGradientOpacity.AddPoint(90, 0.5)
                    volumeGradientOpacity.AddPoint(100, 1.0)

                    # volumeProperty.SetGradientOpacity(volumeGradientOpacity)  # 设置梯度不透明度效果对比
                    # 设置颜色属性
                    # color = vtkColorTransferFunction()
                    # color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)  # 背景：透明黑色
                    # color.AddRGBPoint(1.0, 0.3, 0.3, 0.8)  # 低值：蓝色
                    # color.AddRGBPoint(100.0, 0.8, 0.3, 0.3)  # 中值：红色
                    # color.AddRGBPoint(200.0, 1.0, 1.0, 1.0)  # 高值：白色 ← 关键修改！
                    # color.AddRGBPoint(400.0, 1.0, 1.0, 1.0)  # 确保高值保持白色

                    # 增加不透明度
                    # compositeOpacity = vtkPiecewiseFunction()
                    # compositeOpacity.AddPoint(0, 0.00)  # 背景完全透明
                    # compositeOpacity.AddPoint(50, 0.2)  # 低值部分透明
                    # compositeOpacity.AddPoint(100, 0.6)  # 中值较不透明
                    # compositeOpacity.AddPoint(150, 0.8)  # 高值更不透明 ← 增加不透明度
                    # compositeOpacity.AddPoint(200, 0.9)
                    #
                    # volumeProperty.SetScalarOpacity(compositeOpacity)
                    # volumeProperty.SetColor(color)

                    color = vtkColorTransferFunction()
                    color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
                    color.AddRGBPoint(64.0, 1.0, 0.52, 0.3)
                    color.AddRGBPoint(109.0, 1.0, 1.0, 1.0)
                    color.AddRGBPoint(220.0, 0.2, 0.2, 0.2)
                    volumeProperty.SetColor(color)

                    volume = vtkVolume()   # 和vtkActor作用一致
                    volume.SetMapper(volumeMapper)
                    volume.SetProperty(volumeProperty)
                    if self.volume_old is not None:
                        self.ren.RemoveViewProp(self.volume_old)
                    self.ren.AddViewProp(volume)
                    self.volume_old = volume
                    # self.volume_path = fname[0]
                    camera = self.ren.GetActiveCamera()
                    c = volume.GetCenter()
                    camera.SetViewUp(0, 0, 1)
                    camera.SetPosition(c[0], c[1] - 800, c[2]-200)
                    camera.SetFocalPoint(c[0], c[1], c[2])
                    camera.Azimuth(30.0)
                    camera.Elevation(30.0)
                    self.iren.Initialize()
                    os.remove(os.path.join(self.parent_file, 'new_img.nii.gz'))
                    self.statusbar.showMessage("The segmentation result has been displayed")
            else:
                self.statusbar.showMessage("Please load a correspronding segmentation result")
        elif self.img is None:
            self.statusbar.showMessage("Please load a image first")

    def clinicf_read(self):
        self.factor1 = self.box1.text()
        self.factor2 = self.box2.text()
        self.factor3 = self.box3.text()
        self.factor4 = self.box4.text()
        self.factor5 = self.box5.text()

        if self.is_number(self.factor1) == True and self.is_number(self.factor2) == True and \
            self.is_number(self.factor3) == True and self.is_number(self.factor4) == True:
            self.show_message_clinicf()
            self.statusbar.showMessage("Factors have been loaded.")
        else:
            self.statusbar.showMessage("Please input data as required.")


    def showmask_path(self, path):
        img = itk.ReadImage(path)
        self.showmask = itk.GetArrayFromImage(img)
        if self.showmask.shape == self.img.shape:
            self.affine = nib.load(self.filename).affine
            self.mask_np = nib.load(path).get_fdata()
            self.img_np = nib.load(self.filename).get_fdata()
            self.new_img = self.mask_np * self.img_np
            self.new_img = nib.Nifti1Image(self.new_img, affine=self.affine)
            nib.save(self.new_img, os.path.join(self.parent_file, 'new_img.nii.gz'))
            if self.prinimg is not None:
                self.printmask(0.5)
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.draw_line_x(self.x_x, self.x_y)
                self.draw_line_y(self.y_x, self.y_y)
                self.draw_line_z(self.z_x, self.z_y)

                reader = vtkNIFTIImageReader()
                reader.SetFileName(os.path.join(self.parent_file, 'new_img.nii.gz'))
                reader.Update()

                volumeMapper = vtkGPUVolumeRayCastMapper()
                volumeMapper.SetInputData(reader.GetOutput())

                volumeProperty = vtkVolumeProperty()
                volumeProperty.SetInterpolationTypeToLinear()  # 设置体绘制的属性设置，决定体绘制的渲染效果
                volumeProperty.ShadeOn()   # 打开或者关闭阴影
                volumeProperty.SetAmbient(0.4)
                volumeProperty.SetDiffuse(0.6)  # 漫反射
                volumeProperty.SetSpecular(0.2)  # 镜面反射
                # 设置不透明度
                compositeOpacity = vtkPiecewiseFunction()
                compositeOpacity.AddPoint(70, 0.00)
                compositeOpacity.AddPoint(90, 0.4)
                compositeOpacity.AddPoint(180, 0.6)
                volumeProperty.SetScalarOpacity(compositeOpacity)  # 设置不透明度

                # 设置梯度不透明属性
                volumeGradientOpacity = vtkPiecewiseFunction()
                volumeGradientOpacity.AddPoint(10, 0.0)
                volumeGradientOpacity.AddPoint(90, 0.5)
                volumeGradientOpacity.AddPoint(100, 1.0)

                # volumeProperty.SetGradientOpacity(volumeGradientOpacity)  # 设置梯度不透明度效果对比
                # 设置颜色属性
                color = vtkColorTransferFunction()
                color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
                color.AddRGBPoint(64.0, 1.0, 0.52, 0.3)
                color.AddRGBPoint(109.0, 1.0, 1.0, 1.0)
                color.AddRGBPoint(220.0, 0.2, 0.2, 0.2)
                volumeProperty.SetColor(color)

                volume = vtkVolume()   # 和vtkActor作用一致
                volume.SetMapper(volumeMapper)
                volume.SetProperty(volumeProperty)
                if self.volume_old is not None:
                    self.ren.RemoveViewProp(self.volume_old)
                self.ren.AddViewProp(volume)
                self.volume_old = volume
                # self.volume_path = fname[0]
                camera = self.ren.GetActiveCamera()
                c = volume.GetCenter()
                camera.SetViewUp(0, 0, 1)
                camera.SetPosition(c[0], c[1] - 800, c[2]-200)
                camera.SetFocalPoint(c[0], c[1], c[2])
                camera.Azimuth(30.0)
                camera.Elevation(30.0)
                self.iren.Initialize()
                os.remove(os.path.join(self.parent_file, 'new_img.nii.gz'))


    def printmask(self, alpha=0.3, num_classes=12,  color_map=None):
        new_prinimg = []
        color_pool = [
            [255, 0, 0],  # 红
            [0, 255, 0],  # 绿
            [0, 0, 255],  # 蓝
            [255, 255, 0],  # 黄
            [255, 0, 255],  # 紫
            [0, 255, 255],  # 青
            [128, 0, 0],  # 暗红
            [0, 128, 0],  # 暗绿
            [0, 0, 128],  # 暗蓝
            [255, 128, 0],  # 橙
            [128, 0, 128],  # 靛
            [128, 128, 0]  # 橄榄
        ]

        if color_map is None:
            color_map = {}
        for class_id in range(num_classes):
            if class_id not in color_map and class_id - 1 < len(color_pool):
                color_map[class_id] = color_pool[class_id - 1]

        for i in range(self.leng_max):
            imgone = self.ori_prinimg[i, ...]
            maskone = self.showmask[(self.leng_max - 1) - i, ...]
            imgone = self.normilize(imgone)
            imgone = np.repeat(np.expand_dims(imgone, axis=-1), 3, axis=-1)
            maskone = maskone.astype(np.int32)

            colored_mask = np.zeros((*maskone.shape, 3), dtype=np.uint8)
            for class_id in range(1, num_classes):  # 跳过背景0
                if class_id in color_map:
                    colored_mask[maskone == class_id] = color_map[class_id]
                else:
                    color_map[class_id] = [np.random.randint(50, 255) for _ in range(3)]
                    colored_mask[maskone == class_id] = color_map[class_id]

            blended = (alpha * colored_mask + (1 - alpha) * imgone).clip(0, 255).astype(np.uint8)
            #blended = (alpha * colored_mask + (1 - alpha) * imgone).astype(np.uint8)

            new_prinimg.append(blended[None, ...])

        self.prinimg = np.concatenate(new_prinimg, axis=0)

    def loadlivermaskf(self):
        fname = QFileDialog.getOpenFileName(self, caption='Load liver mask',
                                            directory='testdata',
                                            filter="Image(*.nii *.nii.gz)")
        if len(fname[1]) != 0:
            img = itk.ReadImage(fname[0])
            livermask = itk.GetArrayFromImage(img)
            if livermask.shape == self.img.shape:
                self.livermask = livermask
                self.statusbar.showMessage("The segmentation result of liver and spleen has been loaded")
                #self.livermask = np.where(self.livermask != 0, 1, 0)
            else:
                self.statusbar.showMessage("Please load a correspronding segmentation result")

    def loadfatmaskf(self):
        fname = QFileDialog.getOpenFileName(self, caption='Load fat mask',
                                            directory='testdata',
                                            filter="Image(*.nii *.nii.gz)")
        if len(fname[1]) != 0:
            img = itk.ReadImage(fname[0])
            self.tumor_path = fname[0]
            mask = itk.GetArrayFromImage(img)
            if mask.shape == self.img.shape:
                self.mask = itk.GetArrayFromImage(img)
                self.statusbar.showMessage("The segmentation result of fat has been loaded")
            else:
                self.statusbar.showMessage("Please load a correspronding segmentation result")

    def loadvasmaskf(self):
        fname = QFileDialog.getOpenFileName(self, caption='Load vascular mask',
                                            directory='testdata',
                                            filter="Image(*.nii *.nii.gz)")
        if len(fname[1]) != 0:
            img = itk.ReadImage(fname[0])
            self.tumor_path = fname[0]
            mask = itk.GetArrayFromImage(img)
            if mask.shape == self.img.shape:
                self.mask = itk.GetArrayFromImage(img)
                self.statusbar.showMessage("The segmentation result of vascular has been loaded")
            else:
                self.statusbar.showMessage("Please load a correspronding segmentation result")

    def showclion(self):
        fname = QFileDialog.getOpenFileName(self, caption='Load factors',
                                            directory='testdata',
                                            filter="CSV(*.csv)")
        if len(fname[1]) != 0:
            df = pd.read_csv(fname[0], encoding='windows-1252')
            data = df.iloc[:, :].values
            data = data[:, 1:]
            self.clion_radiomics = data[:, :]
            self.radiomics_names = df.columns[1:].values
            self.csv_filename = fname[0]
            self.statusbar.showMessage("All factors have been loaded")

    def is_number(self, str):
        try:
            if str=='NaN':
                return False
            float(str)
            return True
        except ValueError:
            return False

    def readnumresult(self):
        se = self.printdim_r.text()
        if self.prinimg is not None and self.prinimg.ndim == 4 and self.heatmaprongqi is not None:
            if self.is_number(se) == True:
                se = float(se)
                if se <= 1 and se >= 0:
                    self.prinimg = self.ori_prinimg
                    self.trans_prinimg(se)
                    self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                    self.draw_line_x(self.x_x, self.x_y)
                    self.draw_line_y(self.y_x, self.y_y)
                    self.draw_line_z(self.z_x, self.z_y)
                    self.statusbar.showMessage(f"The new transparency is:{se}")
                else:
                    self.statusbar.showMessage("Wrong input")
            elif self.is_number(se) == False:
                self.statusbar.showMessage("Wrong input")
        elif self.prinimg is not None and self.prinimg.ndim == 4 and self.showmask is not None:
            if self.is_number(se) == True:
                se = float(se)
                if se <= 1 and se >= 0:
                    self.prinimg = self.ori_prinimg
                    self.printmask(se)
                    self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                    self.draw_line_x(self.x_x, self.x_y)
                    self.draw_line_y(self.y_x, self.y_y)
                    self.draw_line_z(self.z_x, self.z_y)
                else:
                    self.statusbar.showMessage("Wrong input")
            elif self.is_number(se) == False:
                self.statusbar.showMessage("Wrong input")
        elif self.prinimg is None:
            self.statusbar.showMessage("Please import data first")
        else:
            self.statusbar.showMessage("Please import data first")

    def embedinput(self, x):
        for i in range(x.shape[1]):
            x[:, i] = x[:, i] + i * 2
        return x


    def show_message_predict(self):
        QMessageBox.information(self, "Note", "The prediction is about to be made and will take longer if no "
                                              "segmentation results are entered and automatic segmentation is not performed",
                                QMessageBox.Yes)

    def show_message_compute_volume(self):
        QMessageBox.information(self, "Note", "If the mask is not imported in advance and there are no related "
                                              "files in the directory, the segmentation will be performed automatically, "
                                              "which will take a longer time",
                                QMessageBox.Yes)

    def show_message_radiomics(self):
        QMessageBox.information(self, "Note", "This step requires the segmentation result of liver and tumour; "
                                              "if masks are not imported in advance, the segmentation will be performed "
                                              "automatically, which will take longer",
                                QMessageBox.Yes)


    def show_message_clinicf(self):
        QMessageBox.information(self, "Note", "If the CSV file has been read, "
                                              "this operation overwrites the data imported from the CSV file",
                                QMessageBox.Yes)

    def predictf(self):
        if self.clion_radiomics is not None:
            QMessageBox.information(self, "Note", "Start to predict",
                                    QMessageBox.Yes)
            self.statusbar.showMessage("Predicting, please wait a moment.")

            clinical_features = {
                'Age': [float(self.factor1)],
                'Na.B': [float(self.factor2)],
                'ALT.B': [float(self.factor3)],
                'Creatinine.B': [float(self.factor4)],
                'CP-Score': [float(self.factor5)]
            }

            calculated_features = {
                'liver_IQ/spleen_IQ': [self.liver_IQ / self.spleen_IQ if self.spleen_IQ != 0 else np.nan],
                'liver_median/spleen_median': [
                    self.liver_median / self.spleen_median if self.spleen_median != 0 else np.nan],
                'VF_vol/Fat_vol': [self.VF_vol / self.Total_vol if self.Total_vol != 0 else np.nan],
                'MIA_12': [self.area_pv / self.area_sv if self.area_sv != 0 else np.nan],
                'maximum area_3': [self.area_lpv],
                'tortuosity_3': [self.tortuosity_lpv],
                'curvature_2': [self.curvature_sv],
                'curvature_3': [self.curvature_lpv]
            }


            df = pd.read_csv(self.csv_filename)
            case_id = self.csv_filename.split('_Radiomics_Features.csv')[0].split('/')[-1]
            img_row = df[df['ID'] == case_id]

            if not img_row.empty:
                img_features = img_row.drop(columns=['ID']).iloc[0].to_dict()
                img_features = {k: [v] for k, v in img_features.items()}
            else:
                img_features = {}


            all_features = {
                'ID': [self.csv_filename.split('_Radiomics_Features.csv')[0].split('/')[-1]],
                **clinical_features,
                **calculated_features,
                **img_features
            }

            cat26 = ['Na.B', 'Age', 'ALT.B', 'Creatinine.B', 'CP-Score',
                     'liver_IQ/spleen_IQ', 'liver_median/spleen_median',
                     'VF_vol/Fat_vol',
                     'l_wavelet-LHH_glrlm_RunVariance', 'l_wavelet-HHL_glcm_MCC',
                     'l_original_glrlm_ShortRunLowGrayLevelEmphasis',
                     'l_wavelet-HHH_glszm_SmallAreaEmphasis', 'l_log-sigma-1-mm-3D_glcm_Imc1',
                     'm_gradient_glrlm_ShortRunHighGrayLevelEmphasis', 'm_original_glcm_Contrast',
                     'm_wavelet-LHL_glcm_DifferenceAverage',
                     'm_gradient_glcm_Correlation', 'm_wavelet-HHL_glcm_ClusterProminence',
                     'm_wavelet-HHH_firstorder_Mean',
                     'm_wavelet-HLL_glszm_SizeZoneNonUniformityNormalized',
                     'm_gradient_glszm_SmallAreaLowGrayLevelEmphasis',
                     'MIA_12', 'maximum area_3', 'tortuosity_3', 'curvature_2', 'curvature_3'
                     ]

            df = pd.DataFrame(all_features)[['ID'] + cat26]
            X_test = df[cat26].values
            pred = self.model.predict_proba(X_test)[:, 1][0]
            if pred > 0.35:
                self.plotresult.setTextColor(QtCore.Qt.red)
            elif pred <= 0.35:
                self.plotresult.setTextColor(QtCore.Qt.green)
            self.plotresult.setText("{0:0.3f}".format(pred))
            self.statusbar.showMessage("Prediction of overt HE has been done.")

        else:
            self.statusbar.showMessage("Incomplete input data, please input data as required")


    def trans_prinimg(self, alpha):
        new_prinimg = []
        for i in range(self.leng_max):
            camone = self.heatmaprongqi[(self.leng_max-1) - i, ...]
            imgone = self.ori_prinimg[i, ...]
            maskone = self.mask[(self.leng_max-1) - i, ...]
            imgone = self.normilize(imgone)
            imgone = np.repeat(np.expand_dims(imgone, axis=-1), 3, axis=-1)
            maskone_ = np.repeat(np.expand_dims(maskone, axis=-1), 3, axis=-1)
            heatmap = applyColorMap(np.uint8(255 * camone), cv2.COLORMAP_JET)
            cam_img = alpha * heatmap * maskone_ + 1 * imgone
            new_prinimg.append(cam_img[None, :, :, :])
        new_prinimg = np.concatenate(new_prinimg, axis=0)
        self.prinimg = new_prinimg


    def preprocessimg(self):
        img = self.img
        mask = self.mask
        space = self.space
        newresolutionxy = 0.7675
        newresolutionz = 1.0
        rsize = [int(img.shape[0] * space[2] / newresolutionz),
                 int(img.shape[1] * space[1] / newresolutionxy), int(img.shape[2] * space[0] / newresolutionxy)]
        space = (newresolutionxy, newresolutionxy, newresolutionz)
        img = st.resize(img, output_shape=rsize, order=1, mode='constant', clip=False, preserve_range=True)
        mask = st.resize(mask, output_shape=rsize,order=0, mode='constant', clip=False, preserve_range=True)
        img = np.clip(img, -17.0, 201.0)
        img = (img - 99.40078) / 39.392952
        return img, mask, space

    def show_message_liver(self):
        QMessageBox.information(self, "Prompt", "The extraction of the liver and spleen will take some time.",
                                QMessageBox.Yes)

    def show_message_fat(self):
        QMessageBox.information(self, "Propmt", "The extraction of the muscle and fat will take some time.",
                                QMessageBox.Yes)
    def show_message_vas(self):
        QMessageBox.information(self, "Propmt", "The extraction of vascular will take some time.",
                                QMessageBox.Yes)
    def liver_seg(self):
        if self.img is not None:
            self.statusbar.showMessage("Begin to segment liver and spleen, please wait a moment")
            self.show_message_liver()
            readimg = nib.load(self.filename)
            nib.save(readimg, os.path.join('testdata/nnUNet_in', 'IMG_0_0000.nii.gz'))
            # inference.predict_simple.main()
            os.system("nnUNet_predict -i testdata/nnUNet_in -o testdata/liverANDspleen_output  -t 26 -p nnUNetPlansFLARE22Small -f all  "
                      "-tr nnUNetTrainerV2_FLARE_Small -m 3d_fullres --mode fastest --disable_tta")

            self.livsplmask_path = self.filename.split('.nii.gz')[0] + '_livspl.nii.gz'
            self.livermask_path = self.filename.split('.nii.gz')[0] + '_liver.nii.gz'
            livermask = itk.ReadImage(os.path.join('testdata/liverANDspleen_output', 'IMG_0.nii.gz'))
            self.livsplmask = itk.GetArrayFromImage(livermask)
            mask_new = np.zeros_like(self.livsplmask)
            for val in range(1, 4):
                print('val', val)
                position_cc = np.where(self.livsplmask == val)
                new_img = np.zeros_like(self.livsplmask)
                new_img[position_cc] = 1
                src_new = itk.GetImageFromArray(new_img)
                src_new1 = self.max_connected_domain(src_new)
                position_cc2 = np.where(itk.GetArrayFromImage(src_new1) == 1)
                mask_new[position_cc2] = val

            livsplmask_arr = np.where(mask_new == 1, 1, np.where(mask_new == 3, 2, 0)).astype(np.uint8)
            self.new_livsplmask = itk.GetImageFromArray(livsplmask_arr)
            self.new_livsplmask.CopyInformation(livermask)
            itk.WriteImage(self.new_livsplmask, self.livsplmask_path)

            livermask_arr = np.where(mask_new == 1, 1, 0).astype(np.uint8)
            self.new_livermask = itk.GetImageFromArray(livermask_arr)
            self.new_livermask.CopyInformation(livermask)
            itk.WriteImage(self.new_livermask, self.livermask_path)

            self.showmask_path(self.livsplmask_path)
            os.remove(os.path.join('testdata/liverANDspleen_output', 'IMG_0.nii.gz'))
            os.remove(os.path.join('testdata/liverANDspleen_output', 'plans.pkl'))
            os.remove(os.path.join('testdata/nnUNet_in', 'IMG_0_0000.nii.gz'))
            self.statusbar.showMessage("Liver and spleen segmentation has been done")
        else:
            self.statusbar.showMessage('Please load a image first')

    def get_livermask(self):
        if self.img is not None:
            readimg = nib.load(self.filename)
            nib.save(readimg, os.path.join('testdata/nnUNet_in', 'IMG_0_0000.nii.gz'))
            os.system("nnUNet_predict -i testdata/nnUNet_in -o testdata/liverANDspleen_output  -t 26 -p nnUNetPlansFLARE22Small -f all  "
                      "-tr nnUNetTrainerV2_FLARE_Small -m 3d_fullres --mode fastest --disable_tta")
            self.livsplmask_path = self.filename.split('.nii.gz')[0] + '_livspl.nii.gz'
            self.livermask_path = self.filename.split('.nii.gz')[0] + '_liver.nii.gz'
            livermask = itk.ReadImage(os.path.join('testdata/liverANDspleen_output', 'IMG_0.nii.gz'))
            self.livsplmask = itk.GetArrayFromImage(livermask)
            mask_new = np.zeros_like(self.livsplmask)
            for val in range(1, 3):
                position_cc = np.where(self.livsplmask == val)
                new_img = np.zeros_like(self.livsplmask)
                new_img[position_cc] = 1
                src_new = itk.GetImageFromArray(new_img)
                src_new1 = self.max_connected_domain(src_new)
                position_cc2 = np.where(itk.GetArrayFromImage(src_new1) == 1)
                mask_new[position_cc2] = val

            livsplmask_arr = np.where(mask_new == 1, 1, np.where(mask_new == 3, 2, 0))
            self.new_livsplmask = itk.GetImageFromArray(livsplmask_arr)
            self.new_livsplmask.CopyInformation(livermask)
            itk.WriteImage(self.new_livsplmask, self.livsplmask_path)

            livermask_arr = np.where(mask_new == 1, 1, 0)
            self.new_livermask = itk.GetImageFromArray(livermask_arr)
            self.new_livermask.CopyInformation(livermask)
            itk.WriteImage(self.new_livermask, self.livermask_path)

            self.showmask_path(self.livsplmask_path)
            os.remove(os.path.join('testdata/liverANDspleen_output', 'IMG_0.nii.gz'))
            os.remove(os.path.join('testdata/liverANDspleen_output', 'plans.pkl'))
            os.remove(os.path.join('testdata/nnUNet_in', 'IMG_0_0000.nii.gz'))
            self.statusbar.showMessage("Liver and spleen segmentation has been done")

        else:
            self.statusbar.showMessage("Please load a image first")

    def vascular_seg(self):
        if self.img is not None:
            readimg = nib.load(self.filename)
            self.statusbar.showMessage("Start to segment vascular, please wait a moment")
            self.show_message_vas()
            nib.save(readimg, os.path.join('testdata/nnUNet_in', 'IMG_0_0000.nii.gz'))
            os.system("nnUNet_predict -i testdata/nnUNet_in "
                      "-o testdata/vascular_output -t 703 -f all  -m 3d_fullres -tr nnUNetTrainerV2 -chk model_best")

            self.vasmask_path = self.filename.split('.nii.gz')[0] + '_vascular.nii.gz'
            vasmask = itk.ReadImage(os.path.join('testdata/vascular_output', 'IMG_0.nii.gz'))
            self.vasmask = itk.GetArrayFromImage(vasmask)
            self.new_vasmask = itk.GetImageFromArray(self.vasmask)
            self.new_vasmask.CopyInformation(vasmask)
            itk.WriteImage(self.new_vasmask, self.vasmask_path)
            self.showmask_path(self.vasmask_path)
            os.remove(os.path.join('testdata/vascular_output', 'IMG_0.nii.gz'))
            os.remove(os.path.join('testdata/vascular_output', 'plans.pkl'))
            os.remove(os.path.join('testdata/nnUNet_in', 'IMG_0_0000.nii.gz'))
            self.statusbar.showMessage("vascular segmentation has been done")
        else:
            self.statusbar.showMessage('Please load image first')

    def get_vascular(self):
        if self.img is not None:
            readimg = nib.load(self.filename)
            nib.save(readimg, os.path.join('testdata/nnUNet_in', 'IMG_0_0000.nii.gz'))
            os.system("nnUNet_predict -i testdata/nnUNet_in "
                      "-o testdata/vascular_output -t 703 -f all  -m 3d_fullres -tr nnUNetTrainerV2 -chk model_best")

            self.vasmask_path = self.filename.split('.nii.gz')[0] + '_vascular.nii.gz'
            vasmask = itk.ReadImage(os.path.join('testdata/vascular_output', 'IMG_0.nii.gz'))
            self.vasmask = itk.GetArrayFromImage(vasmask)
            self.new_vasmask = itk.GetImageFromArray(self.vasmask)
            self.new_vasmask.CopyInformation(vasmask)
            itk.WriteImage(self.new_vasmask, self.vasmask_path)
            self.showmask_path(self.vasmask_path)
            os.remove(os.path.join('testdata/vascular_output', 'IMG_0.nii.gz'))
            os.remove(os.path.join('testdata/vascular_output', 'plans.pkl'))
            os.remove(os.path.join('testdata/nnUNet_in', 'IMG_0_0000.nii.gz'))

            # self.mask = np.where(self.mask == 1, 3,
            #                      np.where(self.mask == 2, 4,
            #                               np.where(self.mask == 3, 5,
            #                                        np.where(self.mask == 4, 6,
            #                                                 np.where(self.mask == 5, 7, 0)))))
            self.statusbar.showMessage("vascular segmentation has been done")

        else:
            self.statusbar.showMessage("Please load a image first")

    def musfat_seg(self):
        if self.img is not None:
            readimg = nib.load(self.filename)
            self.statusbar.showMessage("Start to segment muscle and fat, please wait a moment")
            self.show_message_fat()
            nib.save(readimg, os.path.join('testdata/nnUNet_in', 'IMG_0_0000.nii.gz'))
            os.system("nnUNet_predict -i testdata/nnUNet_in "
                      "-o testdata/muscleANDfat_output -t 06 -m 2d -f all")
            os.system("nnUNet_predict -i testdata/nnUNet_in "
                      "-o testdata/verte_output -t 57 -m 3d_fullres  -f 0")

            self.musfatmask_path = self.filename.split('.nii.gz')[0] + '_musfat.nii.gz'
            self.vertemask_path = self.filename.split('.nii.gz')[0] + '_verte.nii.gz'
            self.musmask_path = self.filename.split('.nii.gz')[0] + '_mus.nii.gz'
            vertemask = itk.ReadImage(os.path.join('testdata/verte_output', 'IMG_0.nii.gz'))
            self.vertemask = itk.GetArrayFromImage(vertemask)
            self.new_vertemask = itk.GetImageFromArray(self.vertemask)
            self.new_vertemask.CopyInformation(vertemask)
            itk.WriteImage(self.new_vertemask, self.vertemask_path)
            self.vertemask_noarch = self.remove_arch(self.vertemask_path)

            musfatmask = itk.ReadImage(os.path.join('testdata/muscleANDfat_output', 'IMG_0.nii.gz'))
            self.musfatmask = itk.GetArrayFromImage(musfatmask)
            self.new_musfatmask = itk.GetImageFromArray(self.musfatmask)
            self.new_musfatmask.CopyInformation(musfatmask)
            itk.WriteImage(self.new_musfatmask, self.musfatmask_path)
            self.masfat_location(self.vertemask_noarch, self.musfatmask_path)
            mask_arr = itk.GetArrayFromImage(itk.ReadImage(self.musfatmask_path))
            mus_arr = np.zeros_like(mask_arr)
            mus_arr[mask_arr == 3] = 1
            mus_arr[mask_arr == 4] = 1
            self.new_musmask = itk.GetImageFromArray(mus_arr)
            self.new_musmask.CopyInformation(musfatmask)
            itk.WriteImage(self.new_musmask, self.musmask_path)

            self.showmask_path(self.musfatmask_path)

            os.remove(os.path.join('testdata/muscleANDfat_output', 'IMG_0.nii.gz'))
            os.remove(os.path.join('testdata/muscleANDfat_output', 'plans.pkl'))
            os.remove(os.path.join('testdata/nnUNet_in', 'IMG_0_0000.nii.gz'))
            os.remove(os.path.join('testdata/verte_output', 'IMG_0.nii.gz'))
            os.remove(os.path.join('testdata/verte_output', 'plans.pkl'))
            self.statusbar.showMessage("muscle and fat segmentation has been done")
        else:
            self.statusbar.showMessage('Please load image first')

    def get_musfatmask(self):
        if self.img is not None:
            readimg = nib.load(self.filename)
            self.statusbar.showMessage("Start to segment muscle and fat, please wait a moment")
            self.show_message_fat()
            nib.save(readimg, os.path.join('testdata/nnUNet_in', 'IMG_0_0000.nii.gz'))
            os.system("nnUNet_predict -i testdata/nnUNet_in "
                      "-o testdata/muscleANDfat_output -t 06 -m 2d -f all")
            os.system("nnUNet_predict -i testdata/nnUNet_in "
                      "-o testdata/verte_output -t 57 -m 3d_fullres  -f 0")
            self.musfatmask_path = self.filename.split('.nii.gz')[0] + '_musfat.nii.gz'
            self.vertemask_path = self.filename.split('.nii.gz')[0] + '_verte.nii.gz'
            self.musmask_path = self.filename.split('.nii.gz')[0] + '_mus.nii.gz'
            vertemask = itk.ReadImage(os.path.join('testdata/verte_output', 'IMG_0.nii.gz'))
            self.vertemask = itk.GetArrayFromImage(vertemask)
            self.new_vertemask = itk.GetImageFromArray(self.vertemask)
            self.new_vertemask.CopyInformation(vertemask)
            itk.WriteImage(self.new_vertemask, self.vertemask_path)
            self.vertemask_noarch = self.remove_arch(self.vertemask_path)
            musfatmask = itk.ReadImage(os.path.join('testdata/muscleANDfat_output', 'IMG_0.nii.gz'))
            self.musfatmask = itk.GetArrayFromImage(musfatmask)
            self.new_musfatmask = itk.GetImageFromArray(self.musfatmask)
            self.new_musfatmask.CopyInformation(musfatmask)
            itk.WriteImage(self.new_musfatmask, self.musfatmask_path)
            self.masfat_location(self.vertemask_noarch, self.musfatmask_path)
            mask_arr = itk.GetArrayFromImage(itk.ReadImage(self.musfatmask_path))
            mus_arr = np.zeros_like(mask_arr)
            mus_arr[mask_arr == 3] = 1
            mus_arr[mask_arr == 4] = 1
            self.new_musmask = itk.GetImageFromArray(mus_arr)
            self.new_musmask.CopyInformation(musfatmask)
            itk.WriteImage(self.new_musmask, self.musmask_path)
            self.showmask_path(self.musfatmask_path)
            os.remove(os.path.join('testdata/muscleANDfat_output', 'IMG_0.nii.gz'))
            os.remove(os.path.join('testdata/muscleANDfat_output', 'plans.pkl'))
            os.remove(os.path.join('testdata/nnUNet_in', 'IMG_0_0000.nii.gz'))
            os.remove(os.path.join('testdata/verte_output', 'IMG_0.nii.gz'))
            os.remove(os.path.join('testdata/verte_output', 'plans.pkl'))
            self.statusbar.showMessage("muscle and fat segmentation has been done")
        else:
            self.statusbar.showMessage("Please load a image first")

    def extract_liver(self,img, mask):
        wholemask = 0
        if self.livermask is not None:

            self.livermask = np.where(self.livermask != 0, 1, 0)

            if np.prod(self.livermask.shape) != np.prod(img.shape):
                self.livermask = st.resize(self.livermask, output_shape=img.shape, order=0, mode='constant', clip=False, preserve_range=True)
            ind = np.where(mask != 0)[2]
            if int(0.6 * img.shape[2]) < ind.max():
                self.flag = True
                img = img * self.livermask
                img = img[:, :, :ind.max()]
                ind = np.where(img != 0)
                segimg = img[np.min(ind[0]): np.max(ind[0]), np.min(ind[1]): np.max(ind[1]), np.min(ind[2]): np.max(ind[2])]
            else:
                self.flag = False
                oriimg = img
                # orimask = mask
                img = img * self.livermask
                img = img[:, :, :int(0.6 * oriimg.shape[2])]
                ind = np.where(img != 0)
                segimg = img[np.min(ind[0]): np.max(ind[0]), np.min(ind[1]): np.max(ind[1]),
                         np.min(ind[2]): np.max(ind[2])]

            return segimg, ind
        else:
            return False, False


    def gettimes(self, new_shape, input_size, strides):
        num_x = 1 + math.ceil((new_shape[0] - input_size[0]) / strides[0])
        num_y = 1 + math.ceil((new_shape[1] - input_size[1]) / strides[1])
        num_z = 1 + math.ceil((new_shape[2] - input_size[2]) / strides[2])
        # print(CT_origianl.shape, change_spacing_shape, new_shape, num_x, num_y, num_z)
        return num_x, num_y, num_z


    def backward_hook(self, module, grad_in, grad_out):
        self.grad_block.append(grad_out[0].detach())

    def farward_hook(self, module, input, output):
        self.fmap_block.append(output)

    def normilize(self, x):
        maa = x.max()
        mii = x.min()
        x = (x - mii) * 255 / (maa - mii)
        return x

    def cam_show_img3d(self, img, oriimg, mask, orimask,  getind, feature_map, grads, out_dir):
        oH, oW, oL = oriimg.shape
        heatmaprongqi = np.zeros(oriimg.shape, dtype=float)
        ind = np.where(mask != 0)[2]
        if not self.flag:
            cutheatmaprongqi = np.zeros([oH, oW, int(0.6 * oL)], dtype=float)
        else:
            cutheatmaprongqi = np.zeros([oH, oW, ind.max()], dtype=float)

        H, W, L = img.shape
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 4
        grads = grads.reshape([grads.shape[0], -1])  # 5
        weights = np.mean(grads, axis=1)  # 6
        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :, :]  # 7
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = st.resize(cam, (H, W, L), order=1, clip=False, preserve_range=True)

        cutheatmaprongqi[np.min(getind[0]): np.max(getind[0]), np.min(getind[1]): np.max(getind[1]),
        np.min(getind[2]): np.max(getind[2])] = cam

        if not self.flag:
            heatmaprongqi[:, :, :int(0.6 * oL)] = cutheatmaprongqi
        else:
            heatmaprongqi[:, :, :ind.max()] = cutheatmaprongqi


        self.heatmaprongqi = st.resize(heatmaprongqi, output_shape=self.img.shape,  clip=False, preserve_range=True)
        self.trans_prinimg(0.3)

        self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
        self.draw_line_x(self.x_x, self.x_y)
        self.draw_line_y(self.y_x, self.y_y)
        self.draw_line_z(self.z_x, self.z_y)

    def eventFilter(self, source, event):
        if source is self.view1:
            # print(event.type)
            if event.type() == QtCore.QEvent.HoverMove:
                self.face_flage = 1
                return True

        elif source is self.view2:
            # print(event.type)
            if event.type() == QtCore.QEvent.HoverMove:
                self.face_flage = 2
                return True


        elif source is self.view3:
            # print(event.type)
            if event.type() == QtCore.QEvent.HoverMove:
                self.face_flage = 3
                # print('3')
                return True
        else:
            self.face_flage = 0

        return False

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            if self.face_flage == 1 and self.leng_img != -100:
                self.leng_img = self.leng_img + 1
                if self.leng_img >= self.leng_max:
                    self.leng_img = self.leng_max - 1
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.y_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
                self.z_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
                self.draw_line_x(self.x_x, self.x_y)
                self.draw_line_y(self.y_x, self.y_y)
                self.draw_line_z(self.z_x, self.z_y)

            elif self.face_flage == 2 and self.leng_img != -100:
                self.width_img = self.width_img + 1
                if self.width_img >= self.width_max:
                    self.width_img = self.width_max - 1
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.x_y = int(round(self.face_h * (self.width_img / self.width_max), 0))
                self.z_x = int(round(self.face_w * (self.width_img / self.width_max), 0))
                self.draw_line_x(self.x_x, self.x_y)
                self.draw_line_y(self.y_x, self.y_y)
                self.draw_line_z(self.z_x, self.z_y)

            elif self.face_flage == 3 and self.leng_img != -100:
                self.high_img = self.high_img + 1
                if self.high_img >= self.high_max:
                    self.high_img = self.high_max - 1
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.x_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
                self.y_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
                self.draw_line_x(self.x_x, self.x_y)
                self.draw_line_y(self.y_x, self.y_y)
                self.draw_line_z(self.z_x, self.z_y)

        elif event.angleDelta().y() < 0:
            if self.face_flage == 1 and self.leng_img != -100:
                self.leng_img = self.leng_img - 1
                if self.leng_img < 0:
                    self.leng_img = 0
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.y_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
                self.z_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
                self.draw_line_x(self.x_x, self.x_y)
                self.draw_line_y(self.y_x, self.y_y)
                self.draw_line_z(self.z_x, self.z_y)

            elif self.face_flage == 2 and self.leng_img != -100:
                self.width_img = self.width_img - 1
                if self.width_img < 0:
                    self.width_img = 0
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.x_y = int(round(self.face_h * (self.width_img / self.width_max), 0))
                self.z_x = int(round(self.face_w * (self.width_img / self.width_max), 0))
                self.draw_line_x(self.x_x, self.x_y)
                self.draw_line_y(self.y_x, self.y_y)
                self.draw_line_z(self.z_x, self.z_y)

            elif self.face_flage == 3 and self.leng_img != -100:
                self.high_img = self.high_img - 1
                if self.high_img < 0:
                    self.high_img = 0
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.x_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
                self.y_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
                self.draw_line_x(self.x_x, self.x_y)
                self.draw_line_y(self.y_x, self.y_y)
                self.draw_line_z(self.z_x, self.z_y)


if __name__ == '__main__':
    # gpu_id = "4"
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    #app.setStyleSheet(app.styleSheet() + "QWidget { background-color: #1f3d1f; }")
    #app.setStyleSheet(app.styleSheet() + "QWidget { background-color: #536445; }")
    #app.setStyleSheet(app.styleSheet() + "QWidget { background-color: #3B6C69; }")

    app.setStyleSheet(app.styleSheet() + "QWidget { background-color: #336666 }")
    app.setStyleSheet(app.styleSheet() + "QMenuBar { background-color: #4D8C84; }")
    app.setStyleSheet(app.styleSheet() + "QPushButton { background-color: #6B9C96; }")
    app.setStyleSheet(app.styleSheet() + "QComboBox { border-color: #6B9C96; }")
    app.setStyleSheet(app.styleSheet() + "QLineEdit { border-color: #6B9C96; }")
    app.setStyleSheet(app.styleSheet() + "QStatusBar { border-color: #6B9C96; }")
    app.setStyleSheet(app.styleSheet() + "QMenuBar { border-color: #336666; }")

    # app.setStyleSheet(app.styleSheet() + "QWidget { background-color: #5F6A74 }")
    # app.setStyleSheet(app.styleSheet() + "QMenuBar { border-color: #5F6A74; }")
    # app.setStyleSheet(app.styleSheet() + "QMenuBar { background-color: #8FA0AE; }")
    # app.setStyleSheet(app.styleSheet() + "QPushButton { background-color: #9FB1C2; }")
    # app.setStyleSheet(app.styleSheet() + "QComboBox { border-color: #9FB1C2; }")
    # app.setStyleSheet(app.styleSheet() + "QLineEdit { border-color: #9FB1C2; }")
    # app.setStyleSheet(app.styleSheet() + "QStatusBar { border-color: #9FB1C2; }")

    mw = MyWindow1()
    mw.show()
    sys.exit(app.exec_())