from annoy import AnnoyIndex
import cv2

from database import DBManagement
from extract_feature import  ExtractFeature
import config

import os
import logging
import time

class FaceSearch:
    def __init__(self, dims = 128, metric = 'euclidean'):
        """
            Everytime turn on, load all embedding vector into a tree.
            With 76 images, building time is 0.001s
        """
        self.dims = dims
        self.metric = metric
        self.feature_index = AnnoyIndex(self.dims, metric=self.metric)
        self.db = DBManagement()
        self.extractFeature = ExtractFeature()
        self._build_tree()

    def _build_tree(self, n_trees=18):
        # FIXME: Build annoy tree here.
        """
        n_trees: parameter should be as twice as much the items
        """
        v_arr = self.db.get_features()
        for index, vector in enumerate(v_arr):
            self.feature_index.add_item(index, vector)
        #n_trees = (index + 1) * 2
        self.feature_index.build(n_trees)

    def save_tree(self, path):
        self.feature_index.save(path)

    def load_tree(self, path):
        """
            Can't use this yet
        """
        self.feature_index = AnnoyIndex(self.dims)
        self.feature_index.load(path)

    def search_index_by_vector(self, image, top_n=1):
        # FIXME: Return the clusters and distance
        data = {'success': False}
        try:
            vector = self.extractFeature.extract_feature_insight_face(image)
            #print('Vector: ', vector.shape)
            distances = self.feature_index.get_nns_by_vector(
                vector, top_n, include_distances=True
            )
            ids = self.db.get_ids()
            results = [{'id': ids[a], 'distance': distances[1][i]}
                    for i, a in enumerate(distances[0])]
            """
            If closest distance is beq to config.face_threshold then it failed
            """
            if results[0]['distance'] >= config.face_threshold:
                results[0]['id'] = 'unknown'
            data['success'] = True
            data['results'] = results
            return data
        except:
            return data

if __name__ == '__main__':
     image = cv2.imread("data/test/jisoo.jpg")
     engine = FaceSearch()
     start_time = time.time()
     print(engine.search_index_by_vector(image))
     print("Find: %s seconds\n" % (time.time() - start_time))
