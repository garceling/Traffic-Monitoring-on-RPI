#import the necessary libraries
import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import math
from datetime import datetime



# Initialize the parameters
confThreshold = 0.5             #Confidence threshold
nmsThreshold = 0.4              #Non-maximum suppression threshold
inpWidth = 416                  #Width of network's input image
inpHeight = 416                 #Height of network's input image


classesFile = "coco.names"



# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# Getting the depth sensor's depth scale 
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
found_rgb = False

#Create alignment primitive with depth as its target stream
align_to = rs.stream.depth
align = rs.align(align_to)

#checks if user has compatible camera pluged in with CMOS sensor
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    sys.exit()


#configure stream depth
W=640
H=480

config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)

#configure stream color
if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
# Start streaming
pipeline.start(config)


# function to get the output layer names in the architecture
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] -1] for i in net.getUnconnectedOutLayers()]

# function to draw bounding box on the detected object with class name
def drawPredicted(classId, conf, left, top, right, bottom, frame,x ,y):
    #draws a rectangle over object
    cv2.rectangle(frame, (left,top), (right,bottom), (255,178,50),3)
    #retrieve the depth of a pixel in meters:
    dpt_frame = pipeline.wait_for_frames().get_depth_frame().as_depth_frame()
    distance = dpt_frame.get_distance(x,y)
    #draws circle at the center of the bounding box of the object
    cv2.circle(frame,(x,y),radius=1,color=(0,0,254), thickness=5)
    label = '%.2f' % conf
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s' %(classes[classId])
    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.putText(frame, label,(left,top-5), cv2.FONT_HERSHEY_SIMPLEX,0.75,(57,255,20),2)
    #get the current time
    dateTimeObj=datetime.now()
    #Horizontal Field of View
    HFOV=69
    #Vertical Field of View
    VFOV=42
    #horizontal angle of pixel
    h_angle=float(((x-W/2)/(W/2))*(HFOV/2))
    #vertical angle of pixel
    v_angle=float(((y-H/2)/(H/2))*(VFOV/2))
    #calculate the euclidean angle 
    e_angle=(h_angle*h_angle+v_angle*v_angle)**0.5
    #if the object is on the left side ( where h_angle>0) then the euclidean angle is negative
    if h_angle>0:
        e_angle=e_angle*(-1)

    distance_string = "Dist: " + str(round(distance,2)) + " m"
    #display the distance on the screen
    cv2.putText(frame,distance_string,(left,top+30), cv2.FONT_HERSHEY_SIMPLEX,0.75,(57,255,20),2)
    angle_string="Angle: " + str(round(e_angle,2)) + " deg"
    #display the angle on the screen
    cv2.putText(frame,angle_string,(left,top+60), cv2.FONT_HERSHEY_SIMPLEX,0.75,(57,255,20),2)
    
    #append the information info into a txt file
    with open(r'/home/pi/Desktop/traffic_output.txt', "a+") as file_object:
        info="Date: " + str(dateTimeObj)+ ", Type: " + label + ", Distance: " + str(round(distance,2)) + " m" + ", Angle: " + str(e_angle) + "\n"
        file_object.write(info)

#Remove the bounding boxes with low confidence using non-maxima suppression
def process_detection(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    #Scan through all the bounding boxes output from the network and keep only the
    #ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0]*frameWidth)
                center_y = int(detection[1]*frameHeight)
                width = int(detection[2]*frameWidth)
                height = int(detection[3]*frameHeight)
                left = int(center_x - width/2)
                top = int(center_y - height/2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left,top,width,height])
                
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        x = int(left+width/2)
        y = int(top+ height/2)
        drawPredicted(classIds[i], confidences[i], left, top, left+width, top+height,frame,x,y)


if __name__ == "__main__":
    classes = None
    # Load names of classes
    with open(classesFile, "rt") as f:
        classes = f.read().rstrip('\n').split('\n')
    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = "yolov3-tiny.cfg"
    modelWeights = "yolov3-tiny.weights"
    
    # load our serialized model from disk
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            blob = cv2.dnn.blobFromImage(color_image, 1/255, (inpWidth, inpHeight), [0,0,0],1,crop=False)
            net.setInput(blob)
            outs = net.forward(getOutputsNames(net))
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            process_detection(color_image,outs)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                 interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))
                
            # Show images
            cv2.imshow('Distance Measurement :)', images)
            #breaks loop/video stream if q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        
        # Stop streaming
        pipeline.stop()
