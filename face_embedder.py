
import numpy as np
import cv2
import os
import sys
sys.path.append('./insightface')
import face_model
import argparse
import time

from face_database import FaceDatabase

class FaceEmbedder:
    def __init__(self):
        parser = argparse.ArgumentParser(description='face model test')
        # general
        parser.add_argument('--image-size', default='112,112', help='')
        parser.add_argument('--model', default='insightface/models/model-r100-ii/model,0', help='path to load model.')
        parser.add_argument('--ga-model', default='', help='path to load model.')
        parser.add_argument('--gpu', default=0, type=int, help='gpu id')
        parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
        args = parser.parse_args()
        self.embedder = face_model.FaceModel(args)

        self.face_db = FaceDatabase()
        self.features = []
        self.ids = []

    def get_feature(self, image):
        face = self.embedder.get_input_without_detect(image)
        start = time.time()
        embedding = self.embedder.get_feature(face)
        end = time.time()
        # print ("embedding time: ", end - start)
        return embedding

    def get_feature_from_folder(self,folderPath):

        fileNumber = 0
        for folder in os.listdir(folderPath):
            path = os.path.join(folderPath, folder)
            # print("folder: ", folder)
            # print("path: ", path)
            for fileName in os.listdir(path):
                filePath = os.path.join(path, fileName)
                print ("filePath: ", filePath," - id: ", folder)
                image = cv2.imread(filePath)
                embedding = self.get_feature(image)
                self.features.append(embedding)
                self.ids.append(folder)
                fileNumber = fileNumber + 1

        print("file number: ", fileNumber)
        self.face_db.save_data(self.features, self.ids)


if __name__ == '__main__':
    faceEmbedder = FaceEmbedder()
    faceEmbedder.get_feature_from_folder("data/images")