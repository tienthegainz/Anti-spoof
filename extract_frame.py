import keras
import keras.backend as K
import h5py
from keras.models import load_model
import time
import os
import cv2
import sys
import numpy as np
#from scipy.misc import imresize, imread
os.environ['KERAS_BACKEND'] = 'tensorflow'

sys.path.append('./insightface')
sys.path.append('insightface/deploy')
sys.path.append('insightface/src/common')
# sys.path.append('./RetinaFace')
sys.path.append('./RetinaFace')

from face_search import FaceSearch
from extract_feature import ExtractFeature
from face_detector import FaceDetector

from gaze_tracking import GazeTracking

faceDetector = FaceDetector()

if __name__ == '__main__':
    cap = cv2.VideoCapture('data/test/tien_fake2.mp4')
    #cap = cv2.VideoCapture('data/test/video-1566446293.mp4')
    #cap = cv2.VideoCapture('data/test/video-1566446358.mp4')
    # Check if camera opened successfully
    count = -1
    name = 'tien1'
    if (cap.isOpened()== False):
      print("Error opening video stream or file")
      exit()
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        count+=1
        if ret == True:
          face_mats = faceDetector.get_face_from_image(frame)
          face_count = 0
          print('{} faces detected in fram {}'.format(len(face_mats), count))
          for face_mat in face_mats:
              img_path = 'data/my_data_real_fake/fake/'+name+'_'+str(count)+'_'+str(face_count)+'.png'
              cv2.imwrite(img_path, face_mat)
              face_count += 1
        elif ret == False:
            break
