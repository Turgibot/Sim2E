import numpy as np
import cv2
import pathlib

import os
import time
try:
    import esim_torch
except Exception as e:
    esim_torch = None
import torch

from datetime import datetime as dt
from project.simulation.unity_enums import *

'''
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
    5: Speed - speed multiplication factor [0, 300]/100                      
    6: PosTh - positive threshold (x1000)                                         
    7: NegTh - negative threshold (x1000)                                     
    8: Attach - connect simulation to real robot - 1 attached, 0 not attached 
    9: Record - record data : 1 recording is on, 0 recording is turned off    
    10: Stereo - for stereoscopic vision : 1 else monoscopic 0
    11: Lighting - 0 to 1.5 multiplied by 100
    12: Target - 0 to 6 [cube, sphere, tetrahedron, torus, mug, spinner, capsule]
'''

def visualize_data(shared_data, sim_positions = None, 
                   sim_ee_config = None, dir_name = "spikes_output",
                   use_esim = False, disable_camera = False):
    print(f"Running with use_esim = {use_esim}")
    print(f"Running with disable_camera = {disable_camera}")
    if use_esim and esim_torch is None:
        use_esim = False
        print("Error loading esim_torch, events will not be generated")
    esim = None
    stereo = StereoEnum.MONOSCOPIC
    recording = False
    shape = None
    neg_th = None
    pos_th = None
    spike_frame = None
    dir_name = "spikes_output" if dir_name is None else dir_name
    title = "Sim2E Visualizer"

    while True:
        width = shared_data[UnityDataEnum.WIDTH]
        height = shared_data[UnityDataEnum.HEIGHT]
        image_data = shared_data[UnityDataEnum.IMAGE_DATA]
        depth_data = shared_data[UnityDataEnum.DEPTH_DATA]
        timestamp = shared_data[UnityDataEnum.TIMESTAMP]

        # Load shared params
        if type(shared_data[UnityDataEnum.PARAMS]) == list:
            try:
                params = list(shared_data[UnityDataEnum.PARAMS])
            except Exception as e:
                print(f"Exception loading params {e}")
                continue
        else:
            time.sleep(1)
            continue

        # Quit if app closed
        if params[UnityEnum.APP_STATUS] == AppStatusEnum.OFF:
            cv2.destroyAllWindows()
            return

        frame = np.array(list(image_data), dtype = np.uint8)
        frame = get_frame(width, height, frame, params[UnityEnum.STEREO])
        depth_frame = np.array(list(depth_data), dtype = np.uint8)
        depth_frame = get_depth_frame(width, height, depth_frame, params[UnityEnum.STEREO])
        depth_frame_bgr = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if use_esim:
            # ESIM restart due to change in camera settings
            if pos_th != params[UnityEnum.POSTH] or \
                neg_th != params[UnityEnum.NEGTH] or \
                    stereo != params[UnityEnum.STEREO]:
                stereo = params[UnityEnum.STEREO]
                esim = None
                neg_th = params[UnityEnum.NEGTH]
                pos_th = params[UnityEnum.POSTH]
            # Set up esim object
            if esim is None:
                esim = esim_torch.esim_torch.EventSimulator_torch(
                    neg_th / 1000, pos_th / 1000, 1e6)
                img_width = width * 2 if stereo == StereoEnum.STEREOSCOPIC else width
                shape = [height, img_width, 3]
                continue
            # Run esim
            sub_events, spike_frame = apply_esim(esim, image_gray, timestamp, shape)
        else:
            sub_events = {}

        # record the events and the frame 
        if params[UnityEnum.RECORD] == RecordEnum.ON and sub_events is not None:
            # Create new recording folder
            if recording is False:
                recording = True
                record_counter = int(dt.now().timestamp())
                scene_path = os.path.join(dir_name, "%010d" % (record_counter))
                pathlib.Path(scene_path).mkdir(parents=True, exist_ok=True)
                frame_counter = 0
            # Record data to file
            output_path = os.path.join(scene_path, "%010d.npz" % frame_counter)
            np_ee_vector = np.array(sim_ee_config)
            sub_events.update({
                'width': width, 'height': height, 'timestamp': timestamp,
                'thetas': sim_positions, 'img': image_gray, 
                'meta': np.array(params, dtype=np.int32),
                'ee_matrix': np_ee_vector.reshape(2,6)
                })
            # Note: the position of the EE is the first three values
            # Note: the position is in meters, so a factor of 1/100 of params 2-4
            np.savez(output_path, **sub_events)
            frame_counter += 1
        else:
            recording = False

        # Show camera
        if not disable_camera:
            if spike_frame is not None:
                all_frames = cv2.vconcat([frame, depth_frame_bgr, spike_frame])
            else:
                all_frames = cv2.vconcat([frame, depth_frame_bgr])
            cv2.imshow(title, all_frames)
            # This might show error message of "cannot move to thread"
            if cv2.waitKey(1) == 27:
                break
        
    cv2.destroyAllWindows()

def apply_esim(esim, image, timestamp, shape):
    log_image = np.log(image.astype("float32") / 255 + 1e-5)
    log_image = torch.from_numpy(log_image).cuda()
    timestamps_ns = torch.from_numpy(np.array([timestamp],dtype=np.int64)).cuda()
    sub_events = esim.forward(log_image, timestamps_ns[0])
    # for the first image, no events are generated, so this needs to be skipped
    if sub_events is not None:
        sub_events = {k: v.cpu() for k, v in sub_events.items()}    
        spike_frame = render(shape=shape, **sub_events)
    else:
        spike_frame = np.full(shape=shape, fill_value=0, dtype="uint8")
    return sub_events, spike_frame

def render(x, y, t, p, shape):
    img = np.full(shape=shape, fill_value=0, dtype="uint8")
    img[y, x, :] = 0
    img[y, x, p] = 255
    return img

       
def get_frame(width, height, frame, stereo):
    frame = frame.reshape((height*2,width, 3))
    left = frame[:height]
    right = frame[height:]
    
    if stereo:
        
        frame = np.concatenate([left, right], axis=1)
    else:
        frame = left
       
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 0)
    return frame

def get_depth_frame(width, height, frame, stereo):
    frame = frame.reshape((height*2,width, 3))
    left = frame[:height]
    right = frame[height:]
    
    if stereo:
        
        frame = np.concatenate([left, right], axis=1)
    else:
        frame = left
    frame *= 2
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.flip(frame, 0)
    return frame
