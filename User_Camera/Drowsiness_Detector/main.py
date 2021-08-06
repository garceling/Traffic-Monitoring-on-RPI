from parameters import *
from scipy.spatial import distance
from imutils import face_utils as face
import imutils
import time
import dlib
import cv2



def get_max_area_rect(rects):
    if len(rects)==0: return
    areas=[]
    for rect in rects:
        areas.append(rect.area())
    return rects[areas.index(max(areas))]

#computes the eye aspect ratio (ear)
def get_eye_aspect_ratio(eye):
    # eye landmarks (x, y)-coordinates
    vertical_1 = distance.euclidean(eye[1], eye[5])
    vertical_2 = distance.euclidean(eye[2], eye[4])
    horizontal = distance.euclidean(eye[0], eye[3])
    #returns EAR
    return (vertical_1+vertical_2)/(horizontal*2) 

#computes the mouth aspect ratio (mar)
def get_mouth_aspect_ratio(mouth):
    # mouth landmarks (x, y)-coordinates
    horizontal=distance.euclidean(mouth[0],mouth[4])
    vertical=0
    for coord in range(1,4):
        vertical+=distance.euclidean(mouth[coord],mouth[8-coord])
    #return MAR
    return vertical/(horizontal*3) 


# Facial processing
def facial_processing():
    distracton_initlized = False
    eye_initialized      = False
    mouth_initialized    = False

	
    detector    = dlib.get_frontal_face_detector()

    #detector = cv2.CascadeClassifier('OpenCV's Haar cascade')
    #OpenCV's Haar cascade for face detection is faster than
    # dlib's built-in HOG detector, but less accurate
	
    predictor   = dlib.shape_predictor('/home/pi/tf/Front Camera/yawn/shape_predictor_68_face_landmarks.dat')

    ls,le = face.FACIAL_LANDMARKS_IDXS["left_eye"]
    rs,re = face.FACIAL_LANDMARKS_IDXS["right_eye"]

    cap=cv2.VideoCapture(0)

    fps_counter=0
    fps_to_display='initializing...'
    fps_timer=time.time()
    # loop over frames from the video stream
    while True:
        _ , frame=cap.read()
        fps_counter+=1
        frame = cv2.flip(frame, 1)
        if time.time()-fps_timer>=1.0:
            fps_to_display=fps_counter
            fps_timer=time.time()
            fps_counter=0
	#displays framerate on screen
        cv2.putText(frame, "FPS :"+str(fps_to_display), (frame.shape[1]-100, frame.shape[0]-10),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	#resize frame
        frame = imutils.resize(frame, width=900)
	#convert frame to grayscale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)
        rect=get_max_area_rect(rects)

        if rect!=None:

            distracton_initlized=False

            shape = predictor(gray, rect)
            shape = face.shape_to_np(shape)
		
	    # extract the left and right eye coordinates, then use the
	    # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[ls:le]
            rightEye = shape[rs:re]
            leftEAR = get_eye_aspect_ratio(leftEye)
            rightEAR = get_eye_aspect_ratio(rightEye)

            inner_lips=shape[60:68]
            mar=get_mouth_aspect_ratio(inner_lips)

	    
	    # average the eye aspect ratio together for both eyes
            eye_aspect_ratio = (leftEAR + rightEAR) / 2.0

	    # compute the convex hull for the left and right eye, then
	    # visualize each of the eyes, draw bounding boxes around eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)
            lipHull = cv2.convexHull(inner_lips)
            cv2.drawContours(frame, [lipHull], -1, (255, 255, 255), 1)

	    #display EAR on screen
            cv2.putText(frame, "EAR: {:.2f} MAR{:.2f}".format(eye_aspect_ratio,mar), (10, frame.shape[0]-10),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	    #checking if eyes are drooping/almost closed
            if eye_aspect_ratio < EYE_DROWSINESS_THRESHOLD:

                if not eye_initialized:
                    eye_start_time= time.time()
                    eye_initialized=True
		#checking if eyes are drowsy for a sufficient number of frames
                if time.time()-eye_start_time >= EYE_DROWSINESS_INTERVAL:
                    cv2.putText(frame, "YOU ARE DROWSY!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            else:
                eye_initialized=False


	    #checks if user is yawning
            if mar > MOUTH_DROWSINESS_THRESHOLD:

                if not mouth_initialized:
                    mouth_start_time= time.time()
                    mouth_initialized=True
		#checks if the user is yawning for a sufficient number of frames
                if time.time()-mouth_start_time >= MOUTH_DROWSINESS_INTERVAL:
                    cv2.putText(frame, "YOU ARE YAWNING!", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            else:
                mouth_initialized=False
	#if the user's face is not focused on the road, the eyes/mouth features cannot be computed
        else:

            if not distracton_initlized:
                distracton_start_time=time.time()
                distracton_initlized=True
	    #checks if the user's eyes are off the road after a sufficient number of frames
            if time.time()- distracton_start_time> DISTRACTION_INTERVAL:

                cv2.putText(frame, "EYES ON ROAD", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

	#show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(5)&0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__=='__main__':
	facial_processing()
