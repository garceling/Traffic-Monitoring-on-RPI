#import the necessary libaries
from parameters import *
from scipy.spatial import distance
from imutils import face_utils as face
import imutils
import time
import dlib
import cv2
from datetime import datetime

#draw a bounding box over face
def get_max_area_rect(rects):
    # checks to see if a face was not dectected (0)
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
    distracton_initialized = False
    eye_initialized      = False
    mouth_initialized    = False
    normal_initialized   = False
	
    #get face detector and facial landmark predector
    detector    = dlib.get_frontal_face_detector()
    predictor   = dlib.shape_predictor('/home/pi/tflite1/Drowsiness_Detector/shape_predictor_68_face_landmarks.dat')

    # grab the indexes of the facial landmarks for the left and right eye, respectively
    ls,le = face.FACIAL_LANDMARKS_IDXS["left_eye"]
    rs,re = face.FACIAL_LANDMARKS_IDXS["right_eye"]

    #start video stream
    cap=cv2.VideoCapture(0)
    
    #count the fps
    fps_counter=0
    fps_to_display='initializing...'
    fps_timer=time.time()
    # loop over frames from the video stream
    while True:
        _ , frame=cap.read()
        fps_counter+=1
	#flip around y-axis
        frame = cv2.flip(frame, 1)
        if time.time()-fps_timer>=1.0:
            fps_to_display=fps_counter
            fps_timer=time.time()
            fps_counter=0
	#displays framerate on screen
        cv2.putText(frame, "FPS :"+str(fps_to_display), (frame.shape[1]-100, frame.shape[0]-10),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


	#convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	
	# detect faces in the grayscale frame
        rects = detector(gray, 0)
	#draw bounding box on face
        rect=get_max_area_rect(rects)


        if rect!=None:
            #measures the duration the users eyes were off the road
            if distracton_initialized==True:
                interval=time.time()-distracton_start_time
                interval=str(round(interval,3))
		#gets the current date/time
                dateTime= datetime.now()
                distracton_initialized=False
                info="Date: " + str(dateTime) + ", Interval: " + interval + ", Type: Eyes not on road"
                info=info+ "\n"
                if time.time()- distracton_start_time> DISTRACTION_INTERVAL:
		   #stores the info into a txt file
                    with open(r'/home/pi/Desktop/output.txt', "a+") as file_object:
                        file_object.write(info)
			
	    # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face.shape_to_np(shape)
		
	    # extract the left and right eye coordinates, then use the
	    # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[ls:le]
            rightEye = shape[rs:re]
	    #gets the EAR for each eye
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
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    #uncomment the three lines below to help check the validity of the program
                    #dateTimeOBJ=datetime.now()
                    #eye_info="Date: " + str(dateTimeOBJ) + " Interval: " + str(time.time()-eye_start_time ) + " Drowsy"
                    #print(eye_info)

            else:
                #measures the duration where the users eyes were drowsy
                if eye_initialized==True:
                    interval_eye=time.time()-eye_start_time
                    interval_eye=str(round(interval_eye,3))
                    dateTime_eye= datetime.now()
                    eye_initialized=False
                    info_eye="Date: " + str(dateTime_eye) + ", Interval: " + interval_eye + ", Type:Drowsy"
                    info_eye=info_eye+ "\n"
		    ##will only store the info if user eyes close/droop for a sufficient amount of time
                    if time.time()-eye_start_time >= EYE_DROWSINESS_INTERVAL:
			#store info into a txt file
                        with open(r'/home/pi/Desktop/output.txt', "a+") as file_object:
                            file_object.write(info_eye)



	    #checks if user is yawning
            if mar > MOUTH_DROWSINESS_THRESHOLD:

                if not mouth_initialized:
                    mouth_start_time= time.time()
                    mouth_initialized=True
		#checks if the user is yawning for a sufficient number of frames
                if time.time()-mouth_start_time >= MOUTH_DROWSINESS_INTERVAL:
                    cv2.putText(frame, "YOU ARE YAWNING!", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    #uncomment the lines below to check the validity of this program
                    #dateTimeOBJ2=datetime.now()
                    #mouth_info="date: " + str(dateTimeOBJ2) + " Interval: " + str(time.time()-mouth_start_time ) + " Yawning" + " mar " + str(mar)
                    #print(mouth_info)

            else:
                #measures duration of users yawn
                if mouth_initialized==True:
                    interval_mouth=time.time()-mouth_start_time
                    interval_mouth=str(round(interval_mouth,3))
                    dateTime_mouth= datetime.now()
                    mouth_initialized=False
                    info_mouth="Date: " + str(dateTime_mouth) + ", Interval: " + interval_mouth + ", Type:Yawning"
                    info_mouth=info_mouth+ "\n"
		    #will only store the info if user yawns for a sufficient amount of time
                    if time.time()-mouth_start_time >= MOUTH_DROWSINESS_INTERVAL:
			#store into into a txt file
                        with open(r'/home/pi/Desktop/output.txt', "a+") as file_object:
                            file_object.write(info_mouth)


            #checks if the user is focused
            if (eye_initialized==False) & (mouth_initialized==False) & (distracton_initialized==False):

                if not normal_initialized:
                    normal_start_time= time.time()
                    normal_initialized=True

		#checks if the user is focused for a sufficient number of frames
                if time.time()-normal_start_time >= NORMAL_INTERVAL:
                        cv2.putText(frame, "Normal!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        print('Normal')
            else:
                if normal_initialized==True:
                    interval_normal=time.time()-normal_start_time
                    interval_normal=str(round(interval_normal,3))
                    dateTime_normal= datetime.now()
                    normal_initialized=False
                    info_normal="Date: " + str(dateTime_normal) + ", Interval: " + interval_normal+ ", Type:Normal"
                    info_normal=info_normal+ "\n"
		    #will only store the info if user is focused for a sufficient amount of time
                    if time.time()-normal_start_time >= NORMAL_INTERVAL:
                        with open(r'/home/pi/Desktop/output.txt', "a+") as file_object:
                            file_object.write(info_normal)


	#if the user's face is not focused on the road, the eyes/mouth features cannot be computed
        else:
            #I added this here, if the user ever turns their head so that their eyes are no longer on the road
            #but at the same time their eyes were drowsy at first, this resets the timer for the drowsiness detector
            if eye_initialized==True:
                    interval_eye=time.time()-eye_start_time
                    interval_eye=str(round(interval_eye,3))
                    dateTime_eye= datetime.now()
                    eye_initialized=False
                    info_eye="Date: " + str(dateTime_eye) + ", Interval: " + interval_eye + ", Type:Drowsy"
                    info_eye=info_eye+ "\n"
                    if time.time()-eye_start_time >= EYE_DROWSINESS_INTERVAL:
                        with open(r'/home/pi/Desktop/output.txt', "a+") as file_object:
                            file_object.write(info_eye)

            if not distracton_initialized:
                distracton_start_time=time.time()
                distracton_initialized=True
                #eye_initialized=False
	    #checks if the user's eyes are off the road after a sufficient number of frames
            if time.time()- distracton_start_time> DISTRACTION_INTERVAL:
		#displays on screen that the driver's eyes are off the road
                cv2.putText(frame, "PLEASE KEEP EYES ON ROAD", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #uncomment the line below if want to check the validity of the program
                #dateTimeOBJ3=datetime.now()
                #DIST_info="date: " + str(dateTimeOBJ3) + " Interval: " + str(time.time()-distracton_start_time) + " EYES NOT ON ROAD"
                #print(DIST_info)

	#show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(5)&0xFF
	
	# if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
	
    # close all windows and release the capture
    cv2.destroyAllWindows()
    cap.release()


if __name__=='__main__':
	facial_processing()
