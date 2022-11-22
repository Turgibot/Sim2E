import asyncio
from cgitb import handler
from collections import defaultdict
from datetime import date
from email.policy import default
import enum
import logging
from re import S
import time
from turtle import update
from typing import AsyncIterable, Iterable

import grpc
import UnityStreamer_pb2
import UnityStreamer_pb2_grpc
import numpy as np
import cv2
import multiprocessing as mp

import data_handler
from scenes import gui_stream
from RoboticArm import RoboticArm

from datetime import datetime
'''
  int32 width = 1;
  int32 height = 2;
  bytes image_data = 3;
  bytes depth_data = 4;
  int64 timestamp = 5;
  repeated int32 params =6;
'''
'''
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
manager = mp.Manager()
shared_data = manager.list()
shared_params = manager.list()
sim_positions = manager.list()
for i in range(6):
    shared_data.append(-1)
    sim_positions.append(0)

for i in range(13):
    shared_params.append(-1)
    

class UnityStreamerServicer(UnityStreamer_pb2_grpc.UnityStreamerServicer):
    def __init__(self, record_conn):
      self.record_conn = record_conn


    async def StreamData(self, request_iterator: AsyncIterable[
        UnityStreamer_pb2.UnityData], unused_context) -> UnityStreamer_pb2.Received:
    
        async for data in request_iterator:
            
            shared_data[0] = data.width
            shared_data[1] = data.height
            shared_data[2] = data.image_data
            shared_data[3] = data.depth_data
            shared_data[4] = data.timestamp
            shared_data[5] = list(data.params)
            for i, prm in enumerate(shared_data[5]):
                shared_params[i] = prm
            
            if self.record_conn is not None:
                self.record_conn.send(shared_data)
            
            if shared_params[0] == 0:
                exit(0)
                

        return UnityStreamer_pb2.Received(timestamp=datetime.timestamp(datetime.now()))
    

async def serve(servicer) -> None:
    server = grpc.aio.server()
    UnityStreamer_pb2_grpc.add_UnityStreamerServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    await server.wait_for_termination()
    cv2.destroyAllWindows()

 
def start_server(record_conn):
    servicer = UnityStreamerServicer(record_conn)
    logging.basicConfig(level=logging.INFO)
    asyncio.get_event_loop().run_until_complete(serve(servicer))


def start_mujoco(from_build=False, shared_params=None, sim_positions=None):
    gui_stream.run(from_build, shared_params, sim_positions)

def start_real_arm(sim_positions):
    while shared_params[8]<=0 :
        if shared_params[0] == 0:
            return
        pass
    factor = 0
    robotic_arm = RoboticArm()
    nap_configuration = [-0.5*np.pi, -0.6*np.pi, 1*np.pi, 0.5*np.pi, 0.4*np.pi, 0]
    robotic_arm.enable_torque()
    robotic_arm.set_map_from_nap(nap_configuration)
    while shared_params[8]==1 and shared_params[0] != 0:
        factor+=1
        if sim_positions[0] != 0:
            robotic_arm.set_position_from_sim(sim_positions)
    robotic_arm.release_torque()

def start(unity_from_build=True):
    
    p0 = mp.Process(target=start_server, args=(None,))
    p1 = mp.Process(target=start_mujoco , args=(unity_from_build, shared_params, sim_positions))
    p2 = mp.Process(target=data_handler.visualize_data , args=(shared_data,))
    p3 = mp.Process(target=start_real_arm, args=(sim_positions,))
    
    
    p0.start()
    p1.start()
    p2.start()
    p3.start()
    
    p0.join()
    p1.join()
    p2.join()
    p3.join()

if __name__ == "__main__":
    start()

  