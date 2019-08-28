# -*- coding: utf-8 -*-
"""
    fcam.extract feature
    -------

    This module implement extract feature of FCam.

    :copyright 2019 by FTECH team.
"""

import sys
sys.path.append('insightface/deploy')
sys.path.append('insightface/src/common')
import face_model
from imutils import paths
import numpy as np
import argparse
import pickle
import cv2
import os


from database import DBManagement

class ExtractFeature():
    def __init__(self):
        self.load_model()

    def load_model(self):
        self.db = DBManagement()
        # Initialize our lists of extracted facial embeddings and corresponding people names
        self.features = []
        self.ids = []
        #self.genders = []
        #self.ages = []
        #self.clusters = []
        #self.fileNames = []

        parser = argparse.ArgumentParser()
        parser.add_argument('--image-size', default='112,112', help='')
        parser.add_argument('--model', default='insightface/models/model-y1-test2/model,0', help='path to load model.')
        parser.add_argument('--ga-model', default='insightface/models/gamodel-r50/model,0', help='path to load model.')
        parser.add_argument('--gpu', default=0, type=int, help='gpu id')
        parser.add_argument('--det', default=0, type=int,
                            help='mtcnn option, 1 means using R+O, 0 means detect from begining')
        parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
        parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
        args = parser.parse_args()

        # Initialize the faces embedder
        self.model = face_model.FaceModel(args)

    def extract_feature_insight_faces(self, imagePath):

        for classname in os.listdir(imagePath):
            total = 0
            for filename in os.listdir(os.path.join(imagePath, classname)):
                path = os.path.join(imagePath, classname, filename)
                #print(path)
                image = cv2.imread(path)

                face = self.model.get_input(image)
                if face is None:
                    print('No face detected\n')
                    continue
                embedding = self.model.get_feature(face)
                id = '{}_{}'.format(classname, total)
                self.ids.append(id)
                self.features.append(embedding)

                total += 1
                print("[extract_feature_insight_face]: Extract face ", id)

            print(total, " faces embedded")
        # save to output
        self.db.save_data(self.features, self.ids)

    def extract_feature_insight_face(self, image):
        face = self.model.get_input(image)
        if face is None:
            return
        embedding = self.model.get_feature(face)
        return embedding

# extractor = ExtractFeature()
# extractor.extract_feature_insight_faces("data/unknown")
if __name__ == '__main__':
    a = ExtractFeature()
    a.extract_feature_insight_faces(imagePath='data/faces')
