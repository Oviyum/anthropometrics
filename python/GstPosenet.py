import sys
sys.path.append('~/coral/project-posenet')
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstVideo', '1.0')
import cairo
import cv2
import time
from gi.repository import Gst, GObject, GstBase, GstVideo
from contextlib import contextmanager
from ctypes import *
from typing import Tuple
import os
from fractions import Fraction
import threading
import numpy as np
import math
from PIL import Image
_temp = __import__('project-posenet.pose_engine', globals(), locals(), ['PoseEngine'], 0)
PoseEngine = _temp.PoseEngine



Gst.init(None)
FIXED_CAPS = Gst.Caps.from_string('video/x-raw,format=RGB,width=[1,2147483647],height=[1,2147483647]')



class GstPosenet(GstBase.BaseTransform):

    
    
    __gstmetadata__ = ('GstPosenet Python','Transform',
                      'gst-python element that can calculate the height of a person', 'dkl')

    __gsttemplates__ = (Gst.PadTemplate.new("src",
                                           Gst.PadDirection.SRC,
                                           Gst.PadPresence.ALWAYS,
                                           FIXED_CAPS),
                       Gst.PadTemplate.new("sink",
                                           Gst.PadDirection.SINK,
                                           Gst.PadPresence.ALWAYS,
                                           FIXED_CAPS))
    __gproperties__ = {
        "calibration": (str,
                  "calibration",
                  "Calibration file",
                  None,
                  GObject.ParamFlags.READWRITE)
    }
    def __init__(self):
        self.Q = None
        self.width = -1
        self.height = -1
        self.calibration = None
        #path to the tflite posenet model
        self.engine = PoseEngine('python/project-posenet/models/mobilenet/posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite')
        self.max_pose_detections = 20
        self.lcameramtx = None
        self.ldist = None
        self.lrectification = None
        self.lprojection = None
        self.rcameramtx = None
        self.rdist = None
        self.rrectification = None
        self.rprojection = None
        self.avg = 0
        self.alpha = 0.1
        self.var = 0
        #height and width of the posenet model
        self.modelw = 1281
        self.modelh = 721
    def calculate_height(self, lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2):

        points2d = np.array([[lx1, ly1, lx1-rx1, 1], [lx2, ly2, lx2-rx2, 1]], dtype=np.float32).T

        #use the Q matrix to project the 2d coordinates to 3d
        points3d = self.Q.dot(points2d)
        x1 = points3d[0][0]/points3d[3][0]
        y1 = points3d[1][0]/points3d[3][0]
        z1 = points3d[2][0]/points3d[3][0]

        x2 = points3d[0][1]/points3d[3][1]
        y2 = points3d[1][1]/points3d[3][1]
        z2 = points3d[2][1]/points3d[3][1]

        #use 3d coordinates to calculate the distance
        distance = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        return distance
    
    def pose(self, img, posearray, i):
        newimg = Image.fromarray(img, 'RGB')
        newimg = newimg.resize((self.modelw, self.modelh), Image.NEAREST)
        lefteye = None
        leftankle = None
        poses, inference_time = self.engine.DetectPosesInImage(np.uint8(newimg))
        for pose in poses:
            if pose.score < 0.4: continue
            for label, keypoint in pose.keypoints.items():
                cv2.circle(img, (int(keypoint.yx[1]/(self.modelw/(self.width/2))), int(keypoint.yx[0]*(self.height/self.modelh))), 1, 255, 3)
                if label == 'left eye':
                    lefteye = keypoint.yx
                if label == 'left ankle':
                    leftankle = keypoint.yx
        posearray[i] = (lefteye, leftankle)
        return (lefteye, leftankle)

#exponential moving average and moving variance with a weight alpha. outliers which are more than one standard deviation from the mean are clipped to one standard deviation from the mean.
    def ema(self, h):
        if self.avg == 0:
            self.avg = h
        else:
            lim = np.sqrt(self.var)
            if h > self.avg and not self.var == 0:
                h = min(h, self.avg + lim)
            elif h < self.avg and not self.var == 0:
                h = max(h, self.avg - lim)
            self.var = (1-self.alpha)*(self.var + self.alpha*((h - self.avg)**2))
            self.avg = self.alpha*h + (1- self.alpha)*self.avg
        return self.avg

        
    
    def do_set_property(self, prop, value):
        if prop.name == 'calibration':
            self.calibration = value
            self.Q = np.load(self.calibration)["Q"]
            self.lcameramtx = np.load(self.calibration)["lcameramtx"]
            self.ldist = np.load(self.calibration)["ldist"]
            self.lrectification = np.load(self.calibration)["lrectification"]
            self.lprojection = np.load(self.calibration)["lprojection"]
            self.rcameramtx = np.load(self.calibration)["rcameramtx"]
            self.rdist = np.load(self.calibration)["rdist"]
            self.rrectification = np.load(self.calibration)["rrectification"]
            self.rprojection = np.load(self.calibration)["rprojection"]


    def do_set_caps(self, incaps, outcaps):
        struct = incaps.get_structure(0)
        self.width = struct.get_int("width").value
        self.height = struct.get_int("height").value
        return True



    def do_transform_ip(self, buf):
        try:

            (_, info) = buf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)

            A = np.ndarray(buf.get_size(), dtype = np.uint8, buffer = info.data)
            A = A.reshape(self.height, self.width, 3).squeeze()
            #left and right frames are separated from the stereo video
            limg = A[:, :self.width//2 - 1]
            rimg = A[:, self.width//2:]
            img = A
            poses = [None, None]
            #thread to allow left and right pose detection to take place simultaneously.
            t = threading.Thread(target=self.pose, args=(rimg, poses, 1))
            t.start()
            self.pose(limg, poses, 0)
            t.join()

            lpose = poses[0]
            rpose = poses[1]
            #ratios of width and height to model width and height to scale the detected coordinates
            xr = (self.width/2)/self.modelw
            yr = self.height/self.modelh
            if lpose[0] is not None and rpose[0] is not None:
                ([ly1, lx1], [ly2, lx2]) = lpose
                ([ry1, rx1], [ry2, rx2]) = rpose
                lpoints = np.array([[lx1*xr, ly1*yr], [lx2*xr, ly2*yr]], dtype=np.float32)
                rpoints = np.array([[rx1*xr, ry1*yr], [rx2*xr, ry2*yr]], dtype=np.float32)

                #the Q matrix is in the same coordinate system as the undistorted image, not the actual image, so we need to first project the detected coordinates into this undistorted system.
                lpoints = cv2.undistortPoints(lpoints, self.lcameramtx, self.ldist, None, self.lrectification, self.lprojection)
                rpoints = cv2.undistortPoints(rpoints, self.rcameramtx, self.rdist, None, self.rrectification, self.rprojection)
                lx1 = lpoints[0][0][0]
                ly1 = lpoints[0][0][1]
                lx2 = lpoints[1][0][0]
                ly2 = lpoints[1][0][1]
                rx1 = rpoints[0][0][0]
                ry1 = rpoints[0][0][1]
                rx2 = rpoints[1][0][0]
                ry2 = rpoints[1][0][1]
                h = self.calculate_height(lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2)
                print('height:')
                print(self.ema(h))
            else:
                print('Pose not detected')
            buf.unmap(info)
            return Gst.FlowReturn.OK
        except Gst.MapError as e:
            Gst.error("Mapping error: %s" % e)
            return Gst.FlowReturn.ERROR







GObject.type_register(GstPosenet)
__gstelementfactory__ = ("GstPosenet", Gst.Rank.NONE, GstPosenet)

