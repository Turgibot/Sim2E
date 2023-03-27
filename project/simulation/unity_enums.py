#############
# Written by AG
# Adds readability to shared_data and shared_params
#############

'''
This is the data that Unity sends:
    shared_data[0] = data.width
    shared_data[1] = data.height
    shared_data[2] = data.image_data
    shared_data[3] = data.depth_data
    shared_data[4] = data.timestamp
    shared_data[5] = list(data.params)

These are the params that unity sends:
index:  values
    0: AppStatus - 1 unity is on, 0 to turnoff the app                        
    1: RobStatus - 0 sleep, 1 start_config, 2 targeting               
    2: Xpos - x position in cm                                                
    3: Ypos - y position in cm                                                
    4: Zpos - z position in cm (for future support)                           
    5: Speed - speed multiplication factor [50, 300]/100                      
    6: PosTh - positive threshold                                             
    7: NegTh - negative threshold                                             
    8: Attach - connect simulation to real robot - 1 attached, 0 not attached 
    9: Record - record data : 1 recording is on, 0 recording is turned off    
    10: Stereo - for stereoscopic vision : 1 else monoscopic 0
    11: Lighting - 0 to 1.5 multiplied by 100
    12: Target - 0 to 6 [cube, sphere, tetrahedron, torus, mug, spinner, capsule]
'''


class UnityDataEnum:
    WIDTH = 0
    HEIGHT = 1
    IMAGE_DATA = 2
    DEPTH_DATA = 3
    TIMESTAMP = 4
    PARAMS = 5


class UnityEnum:
    APP_STATUS = 0
    ROBOT_STATUS = 1
    XPOS = 2
    YPOS = 3
    ZPOS = 4
    SPEED = 5
    POSTH = 6
    NEGTH = 7
    ATTACH = 8
    RECORD = 9
    STEREO = 10
    LIGHTING = 11
    TARGET = 12


class AppStatusEnum:
    ON = 1
    OFF = 0


class RobotStatusEnum:
    SLEEP = 0
    START_CONFIG = 1
    TARGETING = 2


class AttachEnum:
    NOT_ATTACH = 0
    ATTACH = 1


class RecordEnum:
    OFF = 0
    ON = 1


class StereoEnum:
    MONOSCOPIC = 0
    STEREOSCOPIC = 1
