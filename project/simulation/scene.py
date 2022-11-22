"""
Written by Mr. Guy Tordjman 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ Dec 07 2021

Physical simulation is based the MuJoCo open source simulator (http://www.mujoco.org)

This class contains the static robot 3d object. It collects and interfaces static data to be used in the simulation.
Note that dynamic data is collected via the dynamics.py file

"""
import time
import mujoco_py as mjc
import os
import numpy as np
from .utilities import *

class Mujocoation:
    def __init__(self, path_to_xml, unity):
        self.xml = path_to_xml
        try:
            self.model = mjc.load_model_from_path(self.xml)
        except:
            print("cwd: {}".format(os.getcwd()))
            raise Exception("Mujoco failed to load MJCF file from path {}".format(self.xml))
        self.simulation = mjc.MjSim(self.model)
        self.unity = unity
        if(self.unity is None):
            self.viewer = mjc.MjViewer(self.simulation)
            self.cam = self.viewer.cam
            self.cam.distance = 2
            self.cam.azimuth = -90
            self.cam.elevation = -2
            self.cam.lookat[:] = [0, 0, 0.2]



    # This method is for testing purpose only
    # Show the simulation current status. No step incermenting! 
    
    def advance_once(self):
        while True:
            self.add_arrows()
            self.viewer.render()
    
    def show_step(self):
        self.simulation.step()
        if(self.unity is None):
            self.add_arrows()
            self.viewer.render()
        else:
            qpos = self.simulation.data.qpos
            # send data to unity
            # pos = numpy.ndarray(3*nmocap), quat = numpy.ndarray(4*nmocap)
            moc_pos = np.array(self.simulation.data.mocap_pos)
            moc_pos = moc_pos.reshape(len(moc_pos)*3)
            moc_quat = np.array(self.simulation.data.mocap_quat)
            moc_quat = moc_quat.reshape(len(moc_quat)*4)
            self.unity.setqpos(qpos)
            self.unity.setmocap(moc_pos, moc_quat)
            time.sleep(0.001)
    
    def play(self, steps = 10e10):
        counter = 0
        while steps > counter:
            self.simulation.step()
            self.add_arrows()
            self.viewer.render()
            counter += 1
            

    def get_target_pos_euler(self):
        """ Returns the position and orientation of the target """
        
        xyz_target = self.simulation.data.get_body_xpos("target")
        quat_target  = self.simulation.data.get_body_xquat("target")
        euler_angles = euler_from_quaternion(quat_target)
        return np.copy(xyz_target), np.copy(euler_angles)
    
    def get_T_target(self):
        p, _ = self.get_target_pos_euler()
        