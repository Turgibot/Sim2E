"""
This class implements a PID that calculates the 
torque needed by an input of the desired joint angles
"""

"""
Control is achieved by applying torque
Real robot has a builtin PID controller so the input is simply the joint values
"""



from turtle import st
import numpy as np
from .utilities import *
from .kinematics import Kinematics
import sympy as sp

class Control:
    def __init__(self, robot, simulation, theta_d=None) -> None:
        self.robot = robot
        self.model = robot.model
        self.simulation = simulation
        self.kinematics = Kinematics(robot)
        self.phase = 0
        self.theta_d = theta_d if theta_d is not None else np.array(self.robot.home)
        self.final_theta_d = self.theta_d
        self.thetalist= np.array(self.theta_d)
        self.d = np.zeros(self.robot.n_joints)
        self.i = np.zeros(self.robot.n_joints)
        self.prev_err = np.subtract(self.theta_d, self.simulation.data.qpos[:])
        self.kp = 0.5
        self.ki = 0.0
        self.kd = 0.0
        
        
# -----------------------------------------------------------------------------
# FORWARD KINEMATICS
# -----------------------------------------------------------------------------
    
    def FK(self, thetalist=None):
        if thetalist is None:
            thetalist = self.theta_d
        fk = self.kinematics.FK(self.robot.M, self.robot.s_poe, thetalist)
        return fk

   
    #calculate the necessary velocity to drive each joint to a desired theta_d 
    def PID(self, speed=100):
        
        if self.phase == 0:
            self.kp = 1
        elif self.phase == 1:
            self.kp = 5
        else:
            self.kp = 2

        err = np.subtract(self.theta_d, self.simulation.data.qpos[:])
        self.i = np.add(self.i, err)
        self.d = np.subtract(err,  self.prev_err)
        self.prev_err = np.copy(err)
        v = self.kp*err + self.ki*self.i + self.d*self.d
        self.simulation.data.qvel[:] = v*speed/100
            
        u = self.get_gravity_bias()[:]
        self.simulation.data.ctrl[:] = u

    def get_gravity_bias(self):       
        """ Returns the effects of Coriolis, centrifugal, and gravitational forces """
        return self.simulation.data.qfrc_bias[:]

    def trajectoryIK(self, T_target, T_start = None, sections=5, eomg=1e-16, ev=1e-14):
        thetas = self.kinematics.trajectoryIK(T_target, T_start, eomg, ev, sections)
        return thetas
    def CartesianSpaceIK(self, T_target, Tf=5, N=5, method=5):
        htm_list = self.kinematics.CartesianTrajectory(T_target, Tf, N, method)
        return htm_list

    def IK(self, T_target, eomg=1e-16, ev=1e-14):
        thetas = self.kinematics.IK(T_target, eomg, ev)
        return thetas

