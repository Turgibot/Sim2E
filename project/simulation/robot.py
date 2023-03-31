"""
Written by Mr. Guy Tordjman 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ Dec 07 2021

Physical simulation is based the MuJoCo open source simulator (http://www.mujoco.org)

This class is responsible for collecting read-only dynamic data from the robot simulation.
This data is then used to calculate forward and inverse kinematics.

"""
from ntpath import join
import numpy as np
from .utilities import *
import mujoco_py as mjc


class Robot:
    def __init__(self,
                model,      # The mujoco xml generated model
                simulation, # The mujoco simulation object
                home_configuration = [-0.5*np.pi, 0, 0.5*np.pi, 0.5*np.pi, 0.5*np.pi, 0],

                face_down_configuration = [-0.5*np.pi, 0, 0.5*np.pi, 0.5*np.pi, 0, 0],
                
                front_configuration = [-np.pi/2, -0.713, 2.412, np.pi/2, 1.7, 0],
                right_configuration = [-0.64350111, -1.47933899,  1.62545784,  2.50322626,  0.08747117,  0.11719468],
                bottom_right_configuration = [-1.25*np.pi, 0.2*np.pi, 0.75*np.pi, 1.25*np.pi, 1.1*np.pi, 0],
                bottom_left_configuration = [0.25*np.pi, 0.2*np.pi, 0.75*np.pi, -0.25*np.pi, 1.1*np.pi, 0],
                left_configuration = [-2.49809155, -1.47933899,  1.62545785,  0.63836639,  0.08747117, -0.11719467],
                nap_configuration = [-0.5*np.pi, -0.6*np.pi, 1*np.pi, 0.5*np.pi, 0.4*np.pi, 0]
                ):
        self.model = model
        self.simulation = simulation
        self.home = np.array(home_configuration)
        self.ee_home = np.array([0, -0.05, 0.46])
        self.face_down = np.array(face_down_configuration)
        self.ee_face_down = np.array([0, 0.36, 0.267])
        self.front = np.array(front_configuration)
        self.ee_front = np.array([0, 0.3, 0.34])
        self.right = np.array(right_configuration)
        self.ee_right = np.array([0.2, 0, 0.5])
        self.bottom_right = np.array(bottom_right_configuration)
        self.ee_bottom_right = np.array([0.187, -0.0446, 0.059])
        self.bottom_left = np.array(bottom_left_configuration)
        self.ee_bottom_left = np.array([-0.187, -0.0446, 0.059])
        self.left = np.array(left_configuration)
        self.ee_left = np.array([-0.2, 0, 0.5])
        self.nap = np.array(nap_configuration)
        self.n_joints = 6
        self.take_a_nap()

        self.thetas = self.simulation.data.qpos 
        self.thetas_dot = self.simulation.data.qvel
        self.ee_config = self.get_ee_config()
        self.torques = self.simulation.data.ctrl
        self.accel = self.simulation.data.qacc

        # links length in meters
        self.base_link = 0.067
        self.l1 = 0.045
        self.offset = 0.06
        self.l2 = 0.301
        self.l3 = 0.2
        self.l4 = 0.104 #in the -y direction when in the zero configuration
        self.ee_link = 0.075  #in the -y direction when in the zero configuration

        # M matrix
        m_x = np.array([0, -1, 0])
        m_y = np.array([0, 0, 1])
        m_z = np.array([-1, 0, 0])
        m_p = np.array([-self.offset, -0.15, self.base_link+self.l1+self.l2+self.l3+self.l4])
        self.M = np.r_[np.c_[m_x,m_y, m_z, m_p], [[0, 0, 0, 1]]]
        
        # angular velocities in the space form, when in the zero configuration
        self.s_w0 = [0, 0, 1]
        self.s_w1 = [0, -1, 0]
        self.s_w2 = [0, -1, 0]
        self.s_w3 = [0, 0, -1]
        self.s_w4 = [-1, 0, 0]
        self.s_w5 = [0, -1, 0]

        # joint position in the space form, in the zero configuration

        self.s_v0 = [0, 0, 0]
        self.s_v1 = [self.base_link+self.l1, 0, 0]
        self.s_v2 = [self.base_link+self.l1+self.l2, 0, self.offset]
        self.s_v3 = [0, -self.offset, 0]
        self.s_v4 = [0, -self.base_link-self.l1-self.l2-self.l3-self.l4, 0]
        self.s_v5 = [self.base_link+self.l1+self.l2+self.l3+self.l4, 0, self.offset]

        #poe in the space form
        self.s_poe = self.get_space_poe()
        
    def get_space_poe(self):
        s0 = np.array(self.s_w0+self.s_v0)
        s1 = np.array(self.s_w1+self.s_v1)
        s2 = np.array(self.s_w2+self.s_v2)
        s3 = np.array(self.s_w3+self.s_v3)
        s4 = np.array(self.s_w4+self.s_v4)
        s5 = np.array(self.s_w5+self.s_v5)
        return np.c_[s0, s1, s2, s3, s4, s5]

    def take_a_nap(self):
        self.simulation.data.qpos[:] = self.nap
        self.simulation.forward()

    def go_home(self):
        self.simulation.data.qpos[:] = self.home
        self.simulation.forward()

    def zero_config(self):
        self.simulation.data.qpos[:] = np.zeros(6)
        self.simulation.forward()

    def read(self):
        self.thetas = self.simulation.data.qpos
        self.thetas_dot = self.simulation.data.qvel
        self.ee_config = self.get_ee_config()
        self.torques = self.simulation.data.ctrl

    def get_ee_position(self):
        """ Retrieve the position of the End Effector (EE) """
        
        return np.copy(self.simulation.data.get_body_xpos('EE'))
    
   
    def get_target(self, target_name='target'):
        """ Returns the position and orientation of the target """
        
        xyz_target = self.simulation.data.get_body_xpos(target_name)
        quat_target  = self.simulation.data.get_body_xquat(target_name)
        euler_angles = euler_from_quaternion(quat_target)
        return np.hstack([np.copy(xyz_target), np.copy(euler_angles)])
    
    def get_ee_config(self):
        """ Returns the position and orientation of the target """
        
        xyz = self.simulation.data.get_body_xpos("EE")
        quat  = self.simulation.data.get_body_xquat("EE")
        euler_angles = euler_from_quaternion(quat)
        return np.hstack([np.copy(xyz), np.copy(euler_angles)])
    
    

    
    def get_gravity_bias(self):       
        """ Returns the effects of Coriolis, centrifugal, and gravitational forces """
        
        joint_dyn_addrs = np.array((list(self.model.joint_dict.keys())))
        g = -1 * self.simulation.data.qfrc_bias[joint_dyn_addrs]
        return g

   
    def get_links_positions(self):
        pos_dict = {}
        names = ['base_link', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'EE', 'zed']
        for name in names:
            pos = self.simulation.data.get_body_xpos(name)
            pos_dict[name]=pos
        return pos_dict

    def get_joints_pos(self):
        self.thetas = self.simulation.data.qpos
        return self.thetas

    




