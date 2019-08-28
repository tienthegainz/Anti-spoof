from scipy.spatial import distance as dist
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

class MouthTracking():
    """docstring forMouthTracking."""

    def __init__(self, MOUTH_AR_THRESH = 0.7):
        self.MOUTH_AR_THRESH = MOUTH_AR_THRESH
        self.mStart = 49
        self.mEnd = 68
        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    def analyze(self, frame):
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(gray, 0)
        try:
            landmarks = self._predictor(frame, faces[0])
            shape = face_utils.shape_to_np(landmarks)
            self.mouth = shape[self.mStart:self.mEnd]
        except Exception as err:
            print(err)
            self.mouth = None

    @staticmethod
    def mouth_aspect_ratio(mouth):
        # compute the euclidean distances between the two sets of
    	# vertical mouth landmarks (x, y)-coordinates
    	A = dist.euclidean(mouth[2], mouth[9]) # 51, 59
    	B = dist.euclidean(mouth[4], mouth[7]) # 53, 57

    	# compute the euclidean distance between the horizontal
    	# mouth landmark (x, y)-coordinates
    	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

    	# compute the mouth aspect ratio
    	mar = (A + B) / (2.0 * C)

    	# return the mouth aspect ratio
    	return mar

    def is_open(self):
        if MouthTracking.mouth_aspect_ratio(self.mouth) > self.MOUTH_AR_THRESH:
            return True
        else:
            return False
