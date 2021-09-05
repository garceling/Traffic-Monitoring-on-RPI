#import the necessary libraries
from gps import *
import time
from datetime import datetime
import pytz, dateutil.parser


running = True

def getPositionData(gps):
    nx = gpsd.next()
    
    if nx['class'] == 'TPV':
        #extract gps info
        latitude = getattr(nx,'lat', "Unknown")
        longitude = getattr(nx,'lon', "Unknown")
        speed=getattr(nx,'speed', "Unknown")

        #change directory to wherever you want to txt file to be located
        with open(r'/home/pi/Desktop/GPS/newoutput.txt', "a+") as file_object:
            # Move read cursor to the start of file.
            file_object.seek(0)
            # If file is not empty then append '\n'
            data = file_object.read(100)
            if len(data) > 0 :
                file_object.write("\n")
            #stores the current date/time
            dateTimeObj=datetime.now()
            info="Time: " + str(dateTimeObj) + " Position: lon = " + str(longitude) + ", lat = " + str(latitude) + "  Speed: " + str(speed*3.6) + " km/h"
            print (info)
            #Append gps info at the end of the file
            file_object.write(info)
            
# Tell gpsd we are ready to receive messages
gpsd = gps(mode=WATCH_ENABLE|WATCH_NEWSTYLE)


try:
    print ("Application started!")
    while running:
        #call function to extract and append GPS data
        getPositionData(gpsd)
        #delay running the program for 1 second
        time.sleep(1)
        
#if the user presses ctrl+c, the program will stop running
except (KeyboardInterrupt):
    running = False
    
