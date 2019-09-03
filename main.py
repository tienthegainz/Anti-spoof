import keras
import keras.backend as K
import h5py
from keras.models import load_model
import time
import os
import cv2
import sys
import numpy as np
import time
import dlib
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
from face_tracking.gaze_tracking import GazeTracking
from face_tracking.open_mouth_detection import MouthTracking

faceDetector = FaceDetector()
faceSearch = FaceSearch()
faceExtractor = ExtractFeature()
gaze = GazeTracking()
#mouth = MouthTracking()
real_fake_model = load_model('quality_models/liveliness.model')
labels = ['fake', 'Need verify', 'real']
action_sequence_text = ['Close mouth', 'Look left', 'Open mouth', 'Close mouth', 'Look Center', 'Look Right']
action_sequence_status = [[-1, 0], [2, 0], [2, 1], [2, 0], [3, 0], [1, 0]]

def predict_real_fake(face):
    # return string of real or fake
    face = cv2.resize(face, (32, 32))
    face = face /255.00
    #print(face)
    face = np.expand_dims(face, axis=0)
    result = real_fake_model.predict(face)[0]
    #print(result)
    if result[0] > 0.5:
        i = 0
    elif result[0] < 0.5 and result[0] > 0.4:
        i = 1
    else:
        i = 2
    return i, labels[i]

def track_eye_mouth(face, dlib_face_box):
    global gaze
    gaze.refresh(face, dlib_face_box)
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    if gaze.is_right():
        text = "Right"
        status = 1
    elif gaze.is_left():
        text = "Left"
        status = 2
    elif gaze.is_center():
        text = "Center"
        status = 3
    else:
        text = "Unknown"
        status = 0

    '''if gaze.is_top():
        text += " Top"
    elif gaze.is_bottom():
        text += " Bottom"
    elif gaze.is_middle():
        text += " Middle"
    else:
        text += ' Unknown'
    '''
    if gaze.is__mouth_open():
        mouth_text = 'Mouth open'
        mouth_status = 1
    else:
        mouth_text = 'Mouth close'
        mouth_status = 0

    return status, text, left_pupil, right_pupil, mouth_status, mouth_text

def track_mouth(face):
    gaze.analyze(face)
    if mouth.is_open():
        text = 'Mouth open'
        status = 1
    else:
        text = 'Mouth close'
        status = 0
    return status, text

def action_command(eye_status, mouth_status, passed_frame, action_num):
    [eye_command, mouth_command] = action_sequence_status[action_num][0:2]
    # if you take too long
    if passed_frame > 20:
        print('Too long. Restarting')
        return False, 0, 0

    if eye_command == -1 or eye_command == eye_status:
        if mouth_command == -1 or mouth_command == mouth_status:
            passed_frame = 0
            action_num += 1
            return True, passed_frame, action_num
    passed_frame += 1
    return True, passed_frame, action_num

def analysis_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if (cap.isOpened()== False):
      print("Error opening video stream or file")
      return None

    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('data/output/output.avi',fourcc, 20.0, (640,480))

    count = -1
    person_name = ""
    status = -1
    #people = dict() #save previous movement
    action_num = 0
    passed_frame = 0
    action_status = True

    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      count+=1
      if ret == True and count%3 == 0:
        people = dict()
        face_mats, face_boxes = faceDetector.get_face_from_image(frame)

        for i in range(len(face_mats)):
            face_mat = face_mats[i]
            face_box = face_boxes[i]
            dlib_face_box = dlib.rectangle(left=int(face_box[0]),top=int(face_box[1]),right=int(face_box[2]),bottom=int(face_box[3]))
            person = faceSearch.search_index_by_vector(face_mat)

            if person['success'] == True:
                person_name = person['results'][0]['id'].split('_')[0]

            # Quality, texture filter
            quality_status, quality_text = predict_real_fake(face_mat)
            # Motion filter
            eye_status, eye_text, left_pupil, right_pupil, mouth_status, mouth_text = track_eye_mouth(frame, dlib_face_box)
            #mouth_status, mouth_text = track_mouth(face_mat)
            people[str(i)] = {'name': person_name, 'eye': eye_text, 'mouth': mouth_text, 'quality': quality_text}


            '''
            cv2.putText(frame, person_name, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, 'Quality status: {}'.format(quality_text),
                        (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, 'Eye status: {}'.format(eye_text),
                        (10, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
            cv2.circle(frame, left_pupil, 4, (0, 255, 0), thickness=1, lineType=8, shift=0)
            cv2.circle(frame, right_pupil, 4, (0, 255, 0), thickness=1, lineType=8, shift=0)
            cv2.putText(frame, 'Mouth status: {}'.format(mouth_text),
                        (10, 120), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)

            # Filter check
            action_status, passed_frame, action_num = action_command(eye_status, mouth_status, passed_frame, action_num)
            if action_num == len(action_sequence_text):
                print('Unlocked')
                cv2.putText(frame, 'Unlocked',
                            (10, 420), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
                cv2.imshow('Frame',frame)
                exit()
            if action_status:
                cv2.putText(frame, 'Please: {}'.format(action_sequence_text[action_num]),
                            (10, 420), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
            else:
                cv2.putText(frame, 'Too long. Restarting',
                            (10, 420), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)

            cv2.imwrite('data/output/{}.png'.format(count), frame)
            '''

            cv2.imshow('Frame',frame)

        print('In frame {}: {}\n'.format(count, people))
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break

      # Break the loop
      elif ret == False:
        break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    return True

def analysis_image(image_path):
    image = cv2.imread(image_path)
    face_mats, _ = faceDetector.get_face_from_image(image)
    print('{} faces detected'.format(len(face_mats)))
    for face_mat in face_mats:
        print('-----------------')
        person = faceSearch.search_index_by_vector(face_mat)
        print(person)
        print(labels[predict_real_fake(face_mat)])
        text, _, _ = track_eye(image)
        print(text)
        print('-----------------')



if __name__ == '__main__':
    analysis_video('data/test/tien_real7.mp4')
