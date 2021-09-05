import os

#change path to where you saved the shape predictor file
shape_predictor_path    = os.path.join('/home/pi/tf/Front Camera/yawn/', 'shape_predictor_68_face_landmarks.dat')

#feel free to customize these thresholds/intervals 

EYE_DROWSINESS_THRESHOLD    = 0.20
EYE_DROWSINESS_INTERVAL     = 2.0
MOUTH_DROWSINESS_THRESHOLD  = 0.37
MOUTH_DROWSINESS_INTERVAL   = 1.0
DISTRACTION_INTERVAL        = 3.0
NORMAL_INTERVAL             = 1.0

