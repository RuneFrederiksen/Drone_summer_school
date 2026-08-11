#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rpi_capture_image_mavlink.py: Triggering image captures based on RC MAVLink messages 

"""

###############################################
# Definitions                                 #
###############################################
RC_CAPTURE = 6 # RX Channel 7

# Exposure min/max supported for Cam v2: 75, 11766829
CAM_EXPOSURE_TIME = 20000 #in micro seconds, 1000 microsecond => 1ms
CAM_GAIN = 1.0 # analogue gain
CAM_AE = False # Auto Exposure setting. If this is enabled the manual exposure time and gain will be overridden 
CAM_AWB = True # Auto white balance, in case you do not want to let it auto adjust the white balance

###############################################
# Standard Imports                            #
###############################################
import time
import threading
import os

###############################################
# MAVlink Imports                             #
###############################################
from pymavlink import mavutil


###############################################
# OpenCV Imports                              #
###############################################


###############################################
# RPi Imports                                 #
###############################################
import picamera2
import datetime

###############################################
# Drone Control class                         #
###############################################
class DroneControl:
    def __init__(self, *args):
        self.ns = "DroneControl"
        self.armed = False
        self.mode = None
        self.rc_channels = [0,0,0,0,0,0,0,0,0,0,0,0]
        self.index = 0
        self.debounce = False
        self.uav_position = -1
        self.debug = True
        self.first_image_taken = False

        self.state = "INIT"
        self.gps_loc_filename='gps_locations.log'
        # Make New folder for photos

        
        self.camera = self.init_camera()  # change camera properties within this function

        # initialise MAVLink Connection
        
        self.interface = mavutil.mavlink_connection("/dev/PX4", baud=115200, autoreconnect=True)

        self.interface.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        self.interface.wait_heartbeat()

        self.t_run = threading.Thread(target=self.recv_mavlink)
        self.set_state("RUNNING")
        self.t_run.start()
        self.print_debug(">> MAVlink communication is running (Thread)")
        ## end __init__

    def init_camera(self):
        '''
        Initialise the raspberry pi camera with desired properties. 
        For information on the picamera API, see following URL:
        https://picamera.readthedocs.io/en/release-1.13/api_camera.html#
        '''
        camera = picamera2.Picamera2()

        config = camera.create_still_configuration()
        camera.configure(config)
        camera.set_controls({"ExposureTime": CAM_EXPOSURE_TIME, "AnalogueGain": CAM_GAIN, "AeEnable": CAM_AE, "AwbEnable": CAM_AWB})
        camera.start()
        
        return camera

    def print_debug(self,msg):
        ''' Print a message to the console with the prefix "[DroneControl]: '''
        if self.debug == True:
            print("[{}]: >> {} ".format(self.ns,msg))

    def set_state(self, data):
        self.state = data
        self.print_debug("New State: {}".format(data))

    def capture(self):
        
        if(not self.first_image_taken):
            self.create_folder()
            self.first_image_taken = True


        'captures images and gps locations of said images'
        if not self.debounce:
            img_loc = self.folder_loc+'img_'+str(self.index)
            self.camera.start_and_capture_file(img_loc+'.jpg', show_preview=False)
            msg="Image captured. Saved image as img_{}.jpg".format(self.index)
            self.print_debug(msg)
            
            # save geolocation of image
            try:
                write_drone_location = open(self.folder_loc+self.gps_loc_filename,'a')
                
                output = '{},{:.7f},{:.7f},{:.2f},{}\n'.format(
                    self.index,
                    self.uav_position[0],
                    self.uav_position[1],
                    self.uav_position[2],
                    self.uav_position[3],
                )

                write_drone_location.write(output)
                write_drone_location.close()
            except TypeError:
                self.print_debug('WARN: No GPS fix... skipping file save')
                pass
            self.index += 1

            # self.debounce = True
            time.sleep(5)

    def create_folder(self):
        count = 0
        self.folder_loc = '/home/pi/images/capture_{}/'.format(count)
        
        if not os.path.exists(self.folder_loc):
            os.mkdir(self.folder_loc)
        else:
            while(os.path.exists(self.folder_loc)):
                count+=1
                self.folder_loc = '/home/pi/images/capture_{}/'.format(count)
            os.mkdir(self.folder_loc)
                    
        self.print_debug('Saving media to: {}'.format(self.folder_loc))

    def recv_mavlink(self):
        while self.state == "RUNNING":
            try:
                m = self.interface.recv_msg()
            except Exception as e:
                self.debug = False
                print("Lost Connection. Device has possibly been rebooted. Trying to reconnect ...")
                try:
                    self.interface = mavutil.mavlink_connection("/dev/PX4", baud=115200, autoreconnect=True)
                    self.interface.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                    self.interface.wait_heartbeat()
                except:
                    pass
                time.sleep(1)


            #print(m)
            if m:
                self.debug = True
                #print(m)
                if(m.get_type() == 'HEARTBEAT'):
                    #print(m)
                    self.armed = self.decode_armed(m.base_mode)
                    self.mode = self.decode_custom_flightmode(m.custom_mode)
                elif(m.get_type() == 'RC_CHANNELS'):
                    #self.print_debug(m.chan1_raw)
                    self.rc_channels = [m.chan1_raw, # Throttle
                                     m.chan2_raw, # Roll
                                     m.chan3_raw, # Pitch
                                     m.chan4_raw, # Yaw
                                     m.chan5_raw,
                                     m.chan6_raw,
                                     m.chan7_raw,
                                     m.chan8_raw,
                                     m.chan9_raw,
                                     m.chan10_raw,
                                     m.chan11_raw,
                                     m.chan12_raw]
                elif (m.get_type() == 'GLOBAL_POSITION_INT'):
                    # print(m.lat/1e7, m.lon/1e7, m.alt/1e3)
                    self.uav_position = (
                        m.lat/1e7, m.lon/1e7, m.alt/1e3, int(m.hdg/1e2))
                    #self.print_debug(self.rc_channels)

        self.print_debug(">> MAVlink communication has stopped...")
        self.set_state("STOP")

    def decode_armed(self, base_mode):
        return bool((base_mode & 0x80) >> 7)

    def decode_custom_flightmode(self, custom_mode):
        # get bits for sub and main mode
        sub_mode = (custom_mode >> 24)
        main_mode = (custom_mode >> 16) & 0xFF

        # set default value to none
        main_mode_text = ""
        sub_mode_text = ""

        # make list of modes
        mainmodeList = ["MANUAL", "ALTITUDE CONTROL", "POSITION CONTROL", "AUTO", "ACRO", "OFFBOARD", "STABILIZED", "RATTITUDE"]
        submodeList = ["READY", "TAKEOFF", "LOITER", "MISSION", "RTL", "LAND", "RTGS", "FOLLOW_TARGET", "PRECLAND"]

        # get mode from list based on index and what mode we have active. -1 as values is 0-indexed
        # print(main_mode)
        main_mode_text = mainmodeList[main_mode - 1]
        sub_mode_text = submodeList[sub_mode - 1]

        # if no submode, just return mainmode. As a result of -1 will wrap around the array and give precland
        if sub_mode <= 0 or sub_mode > 9:
            return main_mode_text
        else:
            return main_mode_text + " - " + sub_mode_text
    
    def run(self):
        '''
        This is the main loop function
        Add your code in here.
        '''

        # forever loop
        try:
            while True:
                if self.rc_channels[RC_CAPTURE] > 1500:
                    self.capture()
                elif self.debounce:
                    self.debounce = False


                self.print_debug("\033c Testing: \n Armed: {} \n Mode {} \n RC Channels: {} \n".format(
                    self.armed, self.mode, self.rc_channels))
                time.sleep(0.1)
        

        # Kill MAVlink connection when Ctrl-C is pressed (results in a lock)
        except KeyboardInterrupt:
            self.print_debug('CTRL-C has been pressed! Exiting...')
            self.state = "EXIT"
            exit()


if __name__ == '__main__':
    DC = DroneControl()
    DC.print_debug('Brief pause before continuing....')
    time.sleep(1.0)
    DC.run()
