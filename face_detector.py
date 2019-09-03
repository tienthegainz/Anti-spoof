# -*- coding: utf-8 -*-

import sys
sys.path.append('./insightface')
sys.path.append('insightface/deploy')
sys.path.append('insightface/src/common')
import face_preprocess
sys.path.append('./RetinaFace')
from retinaface import RetinaFace


import cv2
import numpy as np
import time
import os


class FaceDetector:
    def __init__(self):
        self.gpuid = -1;
        self.thresh = 0.95;
        self.scales = [2 / 3];
        self.flip = False
        self.detector = RetinaFace('RetinaFace/model/mnet.25', 0, self.gpuid, 'net3')


    def detect(self, image, left_corner_human_box=None):
        start = time.time()
        face_boxes, landmarks = self.detector.detect(image, self.thresh, scales=self.scales, do_flip=self.flip)
        end = time.time()

        for i in range(face_boxes.shape[0]):
            face_boxes[i] = face_boxes[i].astype(np.int)  # x_min, y_min, x_max, y_max
            landmarks[i] = landmarks[i].astype(np.int)  # leye, reye, nose, lmouth, rmouth
            if left_corner_human_box is not None:
                x_human_min = left_corner_human_box[0]
                y_human_min = left_corner_human_box[1]
                # x, y, w, h in orginal coordinate
                face_boxes[i][2] -= face_boxes[i][0]  # w
                face_boxes[i][3] -= face_boxes[i][1]  # h
                face_boxes[i][0] += x_human_min
                face_boxes[i][1] += y_human_min

                for j in range(landmarks[i].shape[0]):
                    landmarks[i][j][0] += x_human_min
                    landmarks[i][j][1] += y_human_min

        return face_boxes[:, :4], landmarks

    def get_faces_from_folder(self,folderPath):
        fileNumber = 0
        for filename in os.listdir(folderPath):
            path = os.path.join(folderPath, filename)
            image = cv2.imread(path)
            face_boxes, landmarks = self.detect(image)
            for i in range(face_boxes.shape[0]):
                faceImage = face_preprocess.preprocess(image, face_boxes[i], landmarks[i], image_size='112,112')
                if not (os.path.exists("data/output")):
                    os.makedirs("data/output")
                filePath = "data/output/" + str(fileNumber) + ".jpg"
                cv2.imwrite(filePath, faceImage)
                fileNumber = fileNumber + 1

    def get_face_from_image(self, image):
        faces = list()
        face_boxes, landmarks = self.detect(image)
        for i in range(face_boxes.shape[0]):
            faceImage = face_preprocess.preprocess(image, face_boxes[i], landmarks[i], image_size='112,112')
            faces.append(faceImage)
        return faces, face_boxes

if __name__ == '__main__':
    faceDetector = FaceDetector()
    faceDetector.get_faces_from_folder("data/test")
