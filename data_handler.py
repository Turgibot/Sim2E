import numpy as np
import cv2
from datetime import datetime
import pickle as pkl
import pathlib

from pytest import param
import os
import esim_torch as esim_torch
import glob
import torch


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

def visualize_data(shared_data):
    frame_counter = 0
    record_counter = 0
    num_events = 0
    esim = None
    prev_gray_frame = None
    spike_frame = None
    prev_spike_frame = None
    stereo = False
    recording = False
    prev_timestamp = None
    shape = None
    img = None
    handle = None
    neg_th = None
    pos_th = None
    dir_name="spikes_output"
    while True:
        width = shared_data[0]
        height = shared_data[1]
        image_data = shared_data[2]
        depth_data = shared_data[3]
        timestamp = shared_data[4]

        try:
            params = list(shared_data[5])
            if pos_th!= params[6]/1000 or neg_th != params[7]/1000 or stereo != params[10]:
                stereo = params[10]
                esim = None

            neg_th = params[7]/1000
            pos_th = params[6]/1000
            frame = np.array(list(image_data), dtype = np.uint8)
            depth_frame = np.array(list(depth_data), dtype = np.uint8)
            if esim is None:
                esim = esim_torch.esim_torch.EventSimulator_torch(neg_th,pos_th,100)
                img_width = width
                if stereo:
                    img_width = width * 2
                
                shape = [height, img_width, 3]
                continue
        except:
            continue
        
        if params[0] == 0:
                cv2.destroyAllWindows()
                return

        frame = get_frame(width, height, frame, params[10])
        depth_frame = get_depth_frame(width, height, depth_frame, params[10])
        depth_frame_bgr = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        log_image = np.log(image.astype("float32") / 255 + 1e-5)
        log_image = torch.from_numpy(log_image).cuda()
        
        timestamps_ns = torch.from_numpy(np.array([timestamp],dtype=np.int64)).cuda()
        sub_events = esim.forward(log_image, timestamps_ns[0])

        # for the first image, no events are generated, so this needs to be skipped
        if sub_events is not None:
            sub_events = {k: v.cpu() for k, v in sub_events.items()}    
            num_events += len(sub_events['t'])
            spike_frame = render(shape=shape, **sub_events)
            all_frames = cv2.vconcat([frame, depth_frame_bgr, spike_frame])
        else:
            all_frames = cv2.vconcat([frame, depth_frame_bgr, np.zeros_like(frame)])

        # record the events and the frame 
        if params[9]==1 and sub_events is not None:
            if recording is False:
                recording = True
                scene_path = os.path.join(dir_name, "%010d" % (record_counter))
                record_counter += 1
                frame_counter = 0
                pathlib.Path(scene_path).mkdir(parents=True, exist_ok=True)

            output_path = os.path.join(scene_path, "%010d.npz" % frame_counter)
            sub_events["img"] = image
            sub_events["meta"] = np.array(params, dtype=np.int32)
            np.savez(output_path, **sub_events)
        else:
            recording = False

        frame_counter += 1

            

        cv2.imshow("", all_frames)
        if cv2.waitKey(1) == 27:
            break
        

    
    cv2.destroyAllWindows()

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

def get_spike_frame(frame, prev_frame, pos_th, neg_th):
    pos_th/=1000
    neg_th/=1000

    src_shape = (frame.shape[0], frame.shape[1], 3) 
    colored_frame = None
    if prev_frame is not None:
        spikes_frame = getSpikesFrom2Frames(prev_frame, frame, pos_th, neg_th).flatten()
        shape = [int(x) for x in spikes_frame.shape]
        colored_frame = np.zeros(shape=shape+[3], dtype="uint8")
        colored_frame[spikes_frame==-1] = [255, 0, 0] 
        colored_frame[spikes_frame==1] = [0, 0, 255] 
        colored_frame = colored_frame.reshape(src_shape)
        
    return frame, colored_frame

def getSpikesFrom2Frames(prev_frame, frame, pos_th, neg_th):
        
    frame = np.array(np.log(frame), dtype=np.float16)
    prev_frame = np.array(np.log(prev_frame), dtype=np.float16)
    deltas = np.array(frame-prev_frame, dtype=np.float16)
    deltas = np.where(deltas >= pos_th, 1, deltas)
    deltas = np.where(deltas <= -neg_th, -1, deltas)
    deltas = np.where(deltas > 1, 0, deltas)
    deltas = np.where(deltas < -1 , 0, deltas)
        
    return deltas
