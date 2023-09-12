import cv2
import numpy as np
import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from GUI import Ui_MainWindow
from PyQt5 import QtWidgets
import tensorflow as tf
import os
import glob
import zipfile
import six.moves.urllib as urllib
import time
from packaging import version
import tarfile
import numpy as np
import multiprocessing


# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util
from PyQt5.QtChart import QChart, QChartView, QPieSeries, QPieSlice
from PyQt5 import QtCore, QtGui, QtWidgets
import csv



os.environ["CUDA_VISIBLE_DEVICES"] = "0 ，1"

with open('traffic_measurement.csv', 'w') as f:
    writer = csv.writer(f)
    csv_line = \
        'Vehicle Type/Size, Vehicle Color, Vehicle Movement Direction, Vehicle Speed (km/h)'
    writer.writerows([csv_line.split(',')])

# MODEL_NAME = 'SSD'
#
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90
speedThreshold = 65
# detection_graph = tf.Graph()
# with detection_graph.as_default():  # 获取当前的默认计算图
#     od_graph_def = tf.GraphDef()  # python Graph中序列化出来的图就叫做 GraphDef
#     with tf.gfile.GFile(PATH_TO_CKPT,
#                         'rb') as fid:  # 获取文件操作，类似于python的open(),PATH_TO_CKPT是打开的文件名，r=read读取数据，b=binary二进制
#         serialized_graph = fid.read()
#         od_graph_def.ParseFromString(serialized_graph)
#         tf.import_graph_def(od_graph_def,
#                             name='')  # 包含要导入到默认图中的操作的od_graph_def proto， name将前缀放在graph_def中名称前面的前缀。注意，这并不适用于导入的函数名。

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                 max_num_classes=NUM_CLASSES,
                                                                 use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class Video():
    def __init__(self, video, window, inference_graph, NUM):
        self.video = video
        self.capture = cv2.VideoCapture(self.video)
        self.window = window
        self.detection_graph = inference_graph


        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.NUM = NUM
        self.total_passed_vehicle = 0
#创建一个video类，其中包括tensorflow 的处理过程和中止过程等
#Create a Video class, including TensorFlow process and terminating operation

    def terminate(self):
        self.capture.release()
        cv2.destroyAllWindows()
        self.window.videoFrame.setText('END OF THE VIDEO FILE')
        self.window.carBar.setValue(0)
        self.window.carBar.setFormat(str(0))
        self.window.truckBar.setValue(0)
        self.window.truckBar.setFormat(str(0))
        self.window.busBar.setValue(0)
        self.window.busBar.setFormat(str(0))




    def tf_process(self):
        total_passed_vehicle = self.total_passed_vehicle
        speed = 'waiting...'
        direction = 'waiting...'
        size = 'waiting...'
        color = 'waiting...'
        #
        carNum = 0
        truckNum = 0
        busNum = 0

        # self.window.label.setText('Speed Threshold:' + str(speedThreshold))


        #
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:  # Session提供了Operation执行和Tensor求值的环境

                # Definite input and output Tensors for detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                for i in range(101):
                    self.window.progressBar.setValue(i)
                    time.sleep(0.05)

                self.window.progressBar.close()

                while self.capture.isOpened():  # 获取视频帧 Get video frame
                    (ret, frame) = self.capture.read()  # ret代表有没有图片，True or False，Frame获取相机下一帧 Ret stands for picture, True or False, Frame gets the camera's next Frame
                    start_time = time.time()


                    if not ret:
                        print('end of the video file...')
                        break

                    input_frame = frame

                    image_np_expanded = np.expand_dims(input_frame, axis=0)  # axis为0，代表矩阵变成了【1，None，None，3】。 Axis is 0, which means the matrix becomes [1, None, None, 3].

                    (boxes, scores, classes, num) = \
                        sess.run([detection_boxes, detection_scores,
                                  detection_classes, num_detections],
                                 feed_dict={image_tensor: image_np_expanded})

                    end_time = time.time() - start_time

                    (counter, csv_line) = \
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            # 有m*n个像素的图片，表示为三维矩阵就是(m, n, 3)，其中m表示高，n表示宽，3表示该元素的RGB色彩值。也就是下面这个矩阵：
                            # An image with m* N pixels, represented as a three-dimensional matrix, is (m, n, 3), where M represents the height, N represents the width, and 3 represents the RGB color value of the element. Which is the following matrix:
                            self.capture.get(1),
                            input_frame,
                            np.squeeze(boxes),  # numpy.squeeze(a,axis = None)，axis为空，删除所有单维度条目,Numpy. squeeze(a,axis = None),axis is empty, delete all single-dimensional entries
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            category_index,
                            self.window.ROI_position,
                            self.window.speed_factor,
                            self.NUM,
                            use_normalized_coordinates=True,
                            line_thickness=4,
                        )

                    total_passed_vehicle = total_passed_vehicle + counter
                    #
                    font = cv2.FONT_HERSHEY_TRIPLEX
                    cv2.putText(
                        input_frame,
                        'Camera No.'+str(self.NUM)+': ' + str(round(end_time,4)),
                        (10, 25),
                        font,
                        0.8,
                        (150, 0xFF, 0xFF),
                        2,
                        cv2.FONT_HERSHEY_SIMPLEX,
                    )
                    cv2.putText(
                        input_frame,
                        'Detected Vehicles: ' + str(total_passed_vehicle),
                        (10, 45),
                        font,
                        0.8,
                        (0, 0xFF, 0xFF),
                        2,
                        cv2.FONT_HERSHEY_SIMPLEX,
                    )
                    cv2.putText(
                        input_frame,
                        'Speed Limit ' + str(speedThreshold)+'km/h',
                        (300, 45),
                        font,
                        0.8,
                        (0, 0xFF, 0xFF),
                        2,
                        cv2.FONT_HERSHEY_SIMPLEX,
                    )

                    ROI_height = int(self.height*self.window.ROI_position)
                    ROI_width = int(self.width)
                    if counter == 1:
                        cv2.line(input_frame, (0, ROI_height), (ROI_width, ROI_height), (0, 0xFF, 0), 5)  # line为Green LINE is Green
                    else:
                        cv2.line(input_frame, (0, ROI_height), (ROI_width, ROI_height), (0, 0, 0xFF), 5)  # line为red Line is Red

                    cv2.rectangle(input_frame, (10, 275), (250, 337), (96, 96, 96), -1)
                    cv2.putText(
                        input_frame,
                        'ROI Line',
                        (545, ROI_height-10),
                        font,
                        0.6,
                        (0, 0, 0xFF),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        input_frame,
                        'LAST PASSED VEHICLE INFO',
                        (11, 290),
                        font,
                        0.5,
                        (0xFF, 0xFF, 0xFF),
                        1,
                        cv2.FONT_HERSHEY_SIMPLEX,
                    )
                    cv2.putText(
                        input_frame,
                        '-Movement Direction: ' + direction,
                        (14, 302),
                        font,
                        0.4,
                        (0xFF, 0xFF, 0xFF),
                        1,
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                    cv2.putText(
                        input_frame,
                        '-Speed(km/h): ' + str(speed).split(".")[0],
                        (14, 312),
                        font,
                        0.4,
                        (0xFF, 0xFF, 0xFF),
                        1,
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                    cv2.putText(
                        input_frame,
                        '-Color: ' + color,
                        (14, 322),
                        font,
                        0.4,
                        (0xFF, 0xFF, 0xFF),
                        1,
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                    cv2.putText(
                        input_frame,
                        '-Vehicle Size/Type: ' + size,
                        (14, 332),
                        font,
                        0.4,
                        (0xFF, 0xFF, 0xFF),
                        1,
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )

                    img = convertFrame(input_frame)
                    self.window.videoFrame.setPixmap(img)
                    self.window.videoFrame.setScaledContents(True)

                    cv2.imshow('vehicle detection', input_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    if csv_line != 'not_available':
                        (size, color, direction, speed) = \
                            csv_line.split(',')
# record whether the vehicle is overspeed
                    try:
                        if counter != 0:
                           if self.video == "input_video.mp4":
                                crop_image = readImg(self.NUM,total_passed_vehicle+1)
                           else: crop_image = readImg(self.NUM,total_passed_vehicle)
                           self.window.label_9.setPixmap(crop_image)
                           self.window.label_9.setScaledContents(True)
                           self.window.label_10.setText('No.'+ str(total_passed_vehicle))
                           if total_passed_vehicle % 4 == 1:
                               self.window.Car1.setPixmap(crop_image)
                               self.window.Car1.setScaledContents(True)
                               self.window.label_1.setText(direction+'\n'
                                                           +str(speed).split(".")[0]+'km/h\n'
                                                           +color+'\n'
                                                           +size+'\n'
                                                           +'No.'+ str(total_passed_vehicle))
                               if speed != 'n.a.':
                                   if int((speed).split(".")[0]) > speedThreshold:
                                       self.window.label_5.setStyleSheet("background-color: rgb(255, 0, 0);")
                                       self.window.label_5.setText('Overspeed')
                                   else:
                                       self.window.label_5.setStyleSheet("background-color: rgb(170, 255, 127);")
                                       self.window.label_5.setText('No\nOverspeed')
                               else :
                                   self.window.label_5.setStyleSheet("background-color: rgb(255, 170, 255);")
                                   self.window.label_5.setText('no\nspeed')
                           if total_passed_vehicle % 4 == 2:
                               self.window.Car2.setPixmap(crop_image)
                               self.window.Car2.setScaledContents(True)
                               self.window.label_2.setText(direction + '\n'
                                                           + str(speed).split(".")[0] + 'km/h\n'
                                                           + color + '\n'
                                                           + size + '\n'
                                                           + 'No.' + str(total_passed_vehicle))
                               if speed != 'n.a.':
                                   if int((speed).split(".")[0]) > speedThreshold:
                                       self.window.label_6.setStyleSheet("background-color: rgb(255, 0, 0);")
                                       self.window.label_6.setText('Overspeed')
                                   else:
                                       self.window.label_6.setStyleSheet("background-color: rgb(170, 255, 127);")
                                       self.window.label_6.setText('No\nOverspeed')
                               else:
                                   self.window.label_6.setStyleSheet("background-color: rgb(255, 170, 255);")
                                   self.window.label_6.setText('no\nspeed')
                           if total_passed_vehicle % 4 == 3:
                               self.window.Car3.setPixmap(crop_image)
                               self.window.Car3.setScaledContents(True)
                               self.window.label_3.setText(direction + '\n'
                                                           + str(speed).split(".")[0] + 'km/h\n'
                                                           + color + '\n'
                                                           + size + '\n'
                                                           + 'No.' + str(total_passed_vehicle))
                               if speed != 'n.a.':
                                   if int((speed).split(".")[0]) > speedThreshold:
                                       self.window.label_7.setStyleSheet("background-color: rgb(255, 0, 0);")
                                       self.window.label_7.setText('Overspeed')
                                   else:
                                       self.window.label_7.setStyleSheet("background-color: rgb(170, 255, 127);")
                                       self.window.label_7.setText('No\nOverspeed')
                               else:
                                   self.window.label_7.setStyleSheet("background-color: rgb(255, 170, 255);")
                                   self.window.label_7.setText('no\nspeed')
                           if total_passed_vehicle % 4 == 0:
                               self.window.Car4.setPixmap(crop_image)
                               self.window.Car4.setScaledContents(True)
                               self.window.label_4.setText(direction + '\n'
                                                           + str(speed).split(".")[0] + 'km/h\n'
                                                           + color + '\n'
                                                           + size + '\n'
                                                           + 'No.' + str(total_passed_vehicle))
                               if speed != 'n.a.':
                                   if int((speed).split(".")[0]) > speedThreshold:
                                       self.window.label_8.setStyleSheet("background-color: rgb(255, 0, 0);")
                                       self.window.label_8.setText('Overspeed')
                                   else:
                                       self.window.label_8.setStyleSheet("background-color: rgb(170, 255, 127);")
                                       self.window.label_8.setText('No\nOverspeed')
                               else:
                                   self.window.label_8.setStyleSheet("background-color: rgb(255, 170, 255);")
                                   self.window.label_8.setText('no\nspeed')

                           if size == 'car':
                               carNum = carNum+ counter
                               self.window.carNum = carNum
                               self.window.carBar.setValue(carNum*10)
                               self.window.carBar.setFormat(str(carNum))
                           elif size == 'truck':
                               truckNum = truckNum + counter
                               self.window.truckNum = truckNum
                               self.window.truckBar.setValue(truckNum*10)
                               self.window.truckBar.setFormat(str(truckNum))
                           elif size == 'bus':
                               busNum = busNum + counter
                               self.window.busNum =busNum
                               self.window.busBar.setValue(busNum*10)
                               self.window.busBar.setFormat(str(busNum))

                           if csv_line != 'not_available':
                               with open('traffic_measurement.csv', 'a') as f:
                                   writer = csv.writer(f)
                                   (size, color, direction, speed) = \
                                       csv_line.split(',')
                                   writer.writerows([csv_line.split(',')])



                    except:
                        return None
            self.capture.release()
            cv2.destroyAllWindows()
            self.window.videoFrame.setText('END OF THE VIDEO FILE')



def convertFrame(frame):
        try:
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine,
                               QImage.Format_RGB888).rgbSwapped()
            qImg = QPixmap.fromImage(qImg)
            return qImg
        except:
            return None


def readImg(NUM,number):
    fileName = "detected_vehicles/"+str(NUM)+"vehicle" + str(number) + ".png"
    crop_Image = QPixmap(fileName)
    return crop_Image


class win(Ui_MainWindow,QtWidgets.QMainWindow):
    def __init__(self):
        super(win,self).__init__()
        self.setupUi(self)
        self.setStyleSheet("background-color: rgb(182, 217, 227);")
        self.setAttribute(Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.videoPath = "input_video.mp4"
        self.MODEL_NAME = 'SSD'
        self.PATH_TO_CKPT = self.MODEL_NAME + '/frozen_inference_graph.pb'
        #
        # self.video = Video(window=self,video = self.videoPath)
        self.progressBar.setValue(0)
        init_image = QPixmap("Traffic.png").scaled(self.width(), self.height())
        self.initialImage.setPixmap(init_image)
        self.initialImage.setScaledContents(True)


        self.NUM = 1

        self.carNum = 0
        self.truckNum = 0
        self.busNum = 0

        self.flag = 1
        self.flag2 = 0


    def change(self):
        self.comboBox.currentText()
        print(self.comboBox.currentText())
        if self.comboBox.currentText() == "Camera No.1":
            self.videoPath = "input_video.mp4"
            self.NUM = 1
        if self.comboBox.currentText() == "Camera No.2":
            self.videoPath = "1.avi"
            self.NUM = 2
        if self.comboBox.currentText() == "Camera No.3":
            self.videoPath = "2.avi"
            self.NUM = 3

        # self.video = Video(window=self, video=self.videoPath)

    def change2(self):
        self.comboBox2.currentText()
        if self.comboBox.currentText() == "SSD" : self.MODEL_NAME ="SSD"
        if self.comboBox.currentText() == "Faster-RCNN" : self.MODEL_NAME = "Faster_RCNN"


    def Test(self):

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.play)
        self._timer.start(27)
        self.update()

    def Graph(self):
        self.pieseries = QPieSeries()  # 定义PieSeries Define the PieSeries
        self.pieseries.append("Car",self.carNum)  # 插入第一个部分 Insert the first part
        self.pieseries.append("Truck", self.truckNum)
        self.pieseries.append("Bus", self.busNum)

        self.slice = self.pieseries.slices()[0]  # 得到饼图的某一个slices Get one of the slice of the pie
        self.slice.setLabelVisible()  # 设置Lable Set the Lable
        self.slice.setPen(QPen(QtCore.Qt.darkGreen, 1))
        self.slice.setBrush(QtCore.Qt.green)

        self.slice2 = self.pieseries.slices()[1]
        self.slice2.setLabelVisible()
        self.slice2.setPen(QPen(QtCore.Qt.darkRed, 1))
        self.slice2.setBrush(QtCore.Qt.red)

        self.slice3 = self.pieseries.slices()[2]
        self.slice3.setLabelVisible()  # 设置Lable
        self.slice3.setPen(QPen(QtCore.Qt.darkBlue, 1))
        self.slice3.setBrush(QtCore.Qt.blue)

        self.chart = QChart()  # 定义QChart
        self.chart.addSeries(self.pieseries)  # 将 pieseries添加到chart里 Add pieseries to chart
        self.chart.setTitle("Overall statistics")  # 设置char的标题 Set the title of char
        self.chart.legend().setVisible(True)

        self.charview = QChartView(self.chart, self)  # 定义charView窗口，插入main UI: Define the charView window and insert main UI
        self.charview.setGeometry(QtCore.QRect(100, 550, 300, 300))  # 设置大小、位置 Set the size and location
        self.charview.setRenderHint(QPainter.Antialiasing)  # 设置抗锯齿 Set anti-aliasing
        if self.flag2 == 0:
            self.charview.show()  # 将CharView窗口显示出来 Show the window
            self.flag2 = 1
            self.pushButton2.close()
            self.pushButton6.show()

    def closeGraph(self):
        self.charview.close()
        self.flag2 = 0
        self.pushButton6.close()
        self.pushButton2.show()

    def stop(self):
            self.videoFrame.close()
            self.flag = 0
            self.pushButton3.close()
            self.pushButton5.show()

    def restart(self):
        self.videoFrame.show()
        self.flag = 1
        self.pushButton5.close()
        self.pushButton3.show()


    def play(self):
        self.flag3 = 1
        self.pushButton.close()
        self.pushButtonT.show()

        self.PATH_TO_CKPT = self.MODEL_NAME + '/frozen_inference_graph.pb'
        detection_graph = tf.Graph()
        with detection_graph.as_default():  # 获取当前的默认计算图   Get the current default computed graph
            od_graph_def = tf.GraphDef()  # python Graph中序列化出来的图就叫做 GraphDef  The Graph serialized from the Python Graph is called GraphDef
            with tf.gfile.GFile(self.PATH_TO_CKPT,
                                'rb') as fid:  # 获取文件操作，类似于python的open(),PATH_TO_CKPT是打开的文件名，r=read读取数据，b=binary二进制 Get the file operation, similar to Python's open(), where PATH_TO_CKPT is the filename opened, r=read reads the data, and b=binary
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def,
                                    name='')  # 包含要导入到默认图中的操作的od_graph_def proto， name将前缀放在graph_def中名称前面的前缀。注意，这并不适用于导入的函数名。
                #The od_graph_def proto that contains the operation to be imported into the default graph, name the prefix that precedes the name in graph_def. Note that this does not apply to imported function names.


        self.video = Video(window=self, video=self.videoPath, inference_graph = detection_graph, NUM = self.NUM)
        self.ROI_position = float(self.textROI.text())
        self.speed_factor = int(self.textSpeed.text())
        self.video.tf_process()


        # except TypeError:
        #     print('No Frame')
    def start(self):
        self.initialImage.close()
        self.pushButton4.close()

    def terminate(self):
         self.video.terminate()
         self.flag3 = 0
         self.pushButtonT.close()
         self.pushButton.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = win()
    win.show()
    sys.exit(app.exec_())