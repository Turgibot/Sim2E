# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=trailing-whitespace
# pylint: disable=bad-whitespace
# pylint: disable=bad-continuation
# pylint: disable=invalid-name

"""
Written by Dr. Elishai Ezra Tsur 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ 7.9.2020

This work is using Robotis' Dynamixel SDK package (pip install dynamixel-sdk) 
and it is utilized here to control Trossen Robotics' arms. 

Code was tested on the: 
1. 5DOF WidowX 200 (https://www.trossenrobotics.com/widowx-200-robot-arm.aspx)
2. 6DOF ViperX 300 (https://www.trossenrobotics.com/viperx-300-robot-arm-6dof.aspx)
"""

import time
import numpy as np
from dynamixel_sdk import *

# Offset angles for the physical arm in relative to the IK mpdel
offset_relative_to_IK_Model = {1: 90, 2: 180, 3: 180, 4: 180, 
                               5: 180, 6: 0, 7: 180, 8: 0, 9: 0}

def robot_to_model_position(robot_position):

    return [     np.deg2rad(robot_position[1]-offset_relative_to_IK_Model[1]),
            -1 * np.deg2rad(robot_position[2]-offset_relative_to_IK_Model[2]),
            -1 * np.deg2rad(robot_position[4]-offset_relative_to_IK_Model[4]),
                 np.deg2rad(robot_position[6]-offset_relative_to_IK_Model[6]),
            -1 * np.deg2rad(robot_position[7]-offset_relative_to_IK_Model[7])]

def model_to_robot_position(model_position):
     
    f = [ np.rad2deg(     model_position[0])+offset_relative_to_IK_Model[1],
          np.rad2deg(-1 * model_position[1])+offset_relative_to_IK_Model[2],
          np.rad2deg(-1 * model_position[2])+offset_relative_to_IK_Model[4],
          np.rad2deg(     model_position[3])+offset_relative_to_IK_Model[6],
          np.rad2deg(-1 * model_position[4])+offset_relative_to_IK_Model[7]]

    return {1: int(f[0]), 2: int(f[1]), 3: int(f[1]), 4: int(f[2]),
            5: int(f[2]), 6: int(f[3]), 7: int(f[4]), 8: 180, 9: 180}

Robot = {'Real':{
            'CMD':
                {'Baud Rate'        : {'Address': 8, 'Value': {9600   : 0,
                                                               57600  : 1,
                                                               115200 : 2,
                                                               1000000: 3,
                                                               2000000: 4,
                                                               3000000: 5,
                                                               4000000: 6,
                                                               4500000: 7}}, 
                 'Operating mode'   : {'Address': 11, 'Value': {'Torque'   : 0,
                                                                'Velocity' : 1,
                                                                'Position' : 3,
                                                                'Extended Position' : 4, 
                                                                'PWM'      : 16}},
                 'Torque Enable'    : {'Address': 64, 'Value': {'OFF': 0, 'ON' : 1}},                           
                 'LED'              : {'Address': 65, 'Value': {'OFF': 0, 'ON' : 1}},                                  
                 'Goal Position'    : {'Address': 116},                         
                 'Present Position' : {'Address': 132}, 
                 'Goal torque'      : {'Address': 102},
                 'Ranges'           : {1: range (-1000, 1000),
                                       2: range (-1000, 1000),
                                       3: range (-1000, 1000),
                                       4: range (-1000, 1000),
                                       5: range (-1000, 1000),
                                       6: range (-1000, 1000),
                                       7: range (-1000, 1000),
                                       8: range (-1000, 1000),
                                       9: range (-1000, 1000)},
                 'Homing Offset'    : {'Address': 20},   
                 'Limit velocity'   : {'Address': 100, 'Value': 600}, # ranging [-885, 885]
                 'Limit torque'     : {'Address': 38,  'Value': 250}  # ranging [-1193, 1193], 2.69mA per step, 3.210A
                },
             'Priority': [[4, 5], [2, 3], [1], [6], [7], [8], [9]],
             
             # Note that engines 2 and 5 were set to reverse mode to allow 
             # both to be configured similarly to their counterpart.
             'Home'     : {1:181, 2:180, 3:180, 4:180, 5:180, 6:0, 7:270, 8:180, 9:180},
             'Nap'      : {1:181, 2:288, 3:72,  4:90,  5:270, 6:0, 7:198, 8:180, 9:180}}
              } 

class RoboticArm:
    
    def __init__ (self, CMD_dict=Robot, COM_ID = '/dev/ttyUSB0', PROTOCOL_VERSION = 2.0):
        
        self.CMD           = CMD_dict['Real']['CMD']
        self.priority      = CMD_dict['Real']['Priority']
        self.home_position = CMD_dict['Real']['Home']
        self.nap = CMD_dict['Real']['Nap']
        self.encoder_unit = 11.37777778

        #map sim to real
        self.id1 = None
        self.id2 = None
        self.id3 = None
        self.id4 = None
        self.id5 = None
        self.id6 = None
        self.id7 = None
        self.id8 = None
        
        self.initialize(COM_ID, PROTOCOL_VERSION)
        
    def initialize(self, COM_ID, PROTOCOL_VERSION):  

        self.portHandler   = PortHandler  (COM_ID)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        # Open port
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")

        # Broadcast ping the Dynamixel
        ENGINES_list, COMM_result = self.packetHandler.broadcastPing(self.portHandler)
        if COMM_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(COMM_result))
        self.ENGINES = []
        print("Detected Engines :")
        for engine in ENGINES_list:
            print("[ID:%03d] model version : %d | firmware version : %d" % 
                  (engine, ENGINES_list.get(engine)[0], ENGINES_list.get(engine)[1]))           
            if ENGINES_list.get(engine)[0] > 1000: # Checking for identified engines (rather then controllers)
                self.ENGINES.append(engine)

        # # Set port baudrate    
        # print('Setting baud rate to: 1 Mbps')
        # for ID in self.ENGINES:
        #     self.send_single_cmd(ID, 
        #                   self.CMD['Baud Rate']['Address'], 
        #                   self.CMD['Baud Rate']['Value'][1000000])
        
        # Initializing mode and constraints
        self.reset_state()
        
        #Initializing syncronized actuation of coupled joints     
        self.groupSyncWrite = GroupSyncWrite(
            self.portHandler, self.packetHandler, self.CMD['Goal Position']['Address'], 4) # 4 for full CMD
        self.groupSyncWrite.clearParam()
     
    def reset_state (self):
    
        # Setting operation mode to position (default; getting into home position)
        # This command has to be excuted first. Changing modes reset default values.       
        print('Releasing torque')
        self.release_torque()
        
        print('Setting operation mode to: position')
        for ID in self.ENGINES:
            if ID == 6:
                self.send_single_cmd(ID, 
                               self.CMD['Operating mode']['Address'], 
                               self.CMD['Operating mode']['Value']['Extended Position'])
                
                self.send_full_cmd(ID, 
                               self.CMD['Homing Offset']['Address'], 
                               round(180*self.encoder_unit))
                
            else:
                self.send_single_cmd(ID, 
                               self.CMD['Operating mode']['Address'], 
                               self.CMD['Operating mode']['Value']['Position'])
            if ID <=5:
                self.send_full_cmd(ID, 
                               84, 
                               500)
            else:
                self.send_full_cmd(ID, 
                               84, 
                               200)
        
        # Limiting velocity
        print('Limiting velocity to: {}%'.format(100*(self.CMD['Limit velocity']['Value']/885)))
        for ID in self.ENGINES:
            self.send_full_cmd(ID, 
                               self.CMD['Limit velocity']['Address'], 
                               self.CMD['Limit velocity']['Value'])
            
        # Limiting torque
        print('Limiting torque to: {}%'.format(100*(self.CMD['Limit torque']['Value']/1193)))
        for ID in self.ENGINES:
            self.send_half_cmd(ID, 
                               self.CMD['Limit torque']['Address'], 
                               self.CMD['Limit torque']['Value'])
   
    def go_home (self) :
        self.enable_torque()
        print("Setting home position")
        self.set_position(self.home_position)
    
    def take_a_nap (self) :
        self.enable_torque()
        print("Setting napping position")
        self.set_position(self.nap)
    
    def reboot (self, IDs = 'all'):
        
        if IDs == 'all':
            IDs = self.ENGINES
        
        for DXL_ID in IDs:
            if DXL_ID < 12: # Assuming less than 12 engines' arm. 
                dxl_comm_result, dxl_error = self.packetHandler.reboot(self.portHandler, DXL_ID)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % self.packetHandler.getRxPacketError(dxl_error))
       
    def enable_torque (self, IDs = 'all'):
        
        if IDs == 'all':
            IDs = self.ENGINES
        
        for DXL_ID in IDs:
            self.send_single_cmd(DXL_ID, 
                          self.CMD['Torque Enable']['Address'], 
                          self.CMD['Torque Enable']['Value']['ON'])  
    
    def release_torque (self, IDs = 'all'):
        
        if IDs == 'all':
            IDs = self.ENGINES
        
        for DXL_ID in IDs:
            self.send_single_cmd(DXL_ID, 
                          self.CMD['Torque Enable']['Address'], 
                          self.CMD['Torque Enable']['Value']['OFF'])
    
    def play_by_priority (self, sequance, delay, mode='position'):
        
        if mode == 'position':
            for seq in sequance:
                self.set_position_by_priority(seq)
                time.sleep(delay)
                
        elif mode == 'torque':
            
            targets_TDs = []
            for seq in sequance:
                for ID in seq:
                    if ID not in targets_TDs:
                        targets_TDs.append(ID)
            # Setting targets to torque control mode
            print('Setting operation mode of engines {} to: torque'.format(targets_TDs))
            for ID in targets_TDs:
                self.release_torque(targets_TDs) # Before changing mode, torque have to be released
                self.send_single_cmd(ID, 
                                   self.CMD['Operating mode']['Address'], 
                                   self.CMD['Operating mode']['Value']['Torque'])
                self.enable_torque(targets_TDs)
            
            for seq in sequance:
                self.set_torque(seq)
                time.sleep(delay)
        else:
            print('Working mode is not recognized. Supported modes: position / torque')
            
    def set_torque (self, torque_dict):
        
        for IDs in self.priority:

            for ID in IDs:
                
                if ID not in torque_dict:
                    continue
                
                target_torque = round(torque_dict[ID])
                print('Setting {} to {}'.format(ID, target_torque))
                self.send_half_cmd(ID, 
                                   self.CMD['Goal torque']['Address'], 
                                   target_torque)
                
    def watch_for_execution(self, ID, target):
        
        watchdog    = time.time()
        lapsed_time = 0

        while True:
            lapsed_time = time.time() - watchdog
            current_position = self.get_position(ID)
            error = round(target-current_position)
            print('{} Deviation is {}'.format(ID, error))
            con = abs(round(target-current_position)) < 500
            if con:
                return True
            if lapsed_time > 2.5:
                print('watch dog executed. ID: {} is at {} instead of {}'.format(ID, current_position, target))

                # Set target to current position. 
                self.send_full_cmd(ID, 
                    self.CMD['Goal Position']['Address'], 
                    self.get_position(ID))
                return False
            
    
    def set_position(self, position_dict):
        
        for ID in position_dict:

            target_position = round(position_dict[ID])
            target_position = round(target_position * self.encoder_unit)
            target_position_byte_array = [DXL_LOBYTE(DXL_LOWORD(target_position)), 
                                          DXL_HIBYTE(DXL_LOWORD(target_position)), 
                                          DXL_LOBYTE(DXL_HIWORD(target_position)), 
                                          DXL_HIBYTE(DXL_HIWORD(target_position))]
            self.groupSyncWrite.addParam(ID, target_position_byte_array)
                
        # print('Setting position at {}'.format(position_dict))
        self.groupSyncWrite.txPacket()

        # for ID in position_dict:
        #     self.watch_for_execution(ID, round(position_dict[ID] * self.encoder_unit)) 
        
        self.groupSyncWrite.clearParam()
    
    def set_position_by_priority (self, position_dict):

        for IDs in self.priority:
            
            if len(IDs) > 1: # Synchronized actuation
                
                for ID in IDs:
                    target_position = round(position_dict[ID] * self.encoder_unit)
                    target_position_byte_array = [DXL_LOBYTE(DXL_LOWORD(target_position)), 
                                                  DXL_HIBYTE(DXL_LOWORD(target_position)), 
                                                  DXL_LOBYTE(DXL_HIWORD(target_position)), 
                                                  DXL_HIBYTE(DXL_HIWORD(target_position))]
                    self.groupSyncWrite.addParam(ID, target_position_byte_array)
                print('Setting {} to {}'.format(IDs, target_position))
                self.groupSyncWrite.txPacket()
                
                for ID in IDs:
                    self.watch_for_execution(ID, target_position) 

            else:
                ID = IDs[0]
                
                if ID not in position_dict:
                    continue
                
                target_position = round(position_dict[ID])
                if target_position not in self.CMD['Ranges'][ID]:
                    print('{} is not in range. Engine {} is constrained to {}'.format(
                        position_dict[ID], ID, self.CMD['Ranges'][ID]))
                    continue

                # Multiplying angle by self.encoder_unit to convert to register value
                target_position = round(target_position * self.encoder_unit)
                print('Setting {} to {}'.format(ID, target_position))

                self.send_full_cmd(ID, 
                              self.CMD['Goal Position']['Address'], 
                              target_position)
            
                self.watch_for_execution(ID, target_position) 
    
    def get_position(self, ID):
        
        dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
            self.portHandler, ID, self.CMD['Present Position']['Address'])
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        return dxl_present_position
    
    def destruct(self):
        self.reboot()
        self.portHandler.closePort()
       
    def send_full_cmd(self, ID, adr, val):
        
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                self.portHandler, ID, adr, val)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            print('ID: {}, Address: {}, value: {}'.format(ID, adr, val))
        # else:
        #     print("[ID:%03d] CMD executed successfully" % ID)
    
    def send_half_cmd(self, ID, adr, val):
        
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
                self.portHandler, ID, adr, val)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            print('ID: {}, Address: {}, value: {}'.format(ID, adr, val))
        # else:
        #     print("[ID:%03d] CMD executed successfully" % ID)
    
    def send_single_cmd(self, ID, adr, val):
        
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, ID, adr, val)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            print('ID: {}, Address: {}, value: {}'.format(ID, adr, val))
        # else:
        #     print("[ID:%03d] CMD executed successfully" % ID)

    def get_positions_in_rad(self):
        res = {}
        for i in range(1, 10):
            j = self.get_position(i)
            res[i] = np.deg2rad(j/self.encoder_unit)
        return res
    
    def set_map_from_nap(self, sim_nap):
        
        real_nap = self.get_positions_in_rad()
        self.id1 = real_nap[1]-sim_nap[0]
        self.id2 = real_nap[2]+sim_nap[1]
        self.id3 = self.id2
        self.id4 = real_nap[4]+sim_nap[2]
        self.id5 = real_nap[5]-sim_nap[2]
        self.id6 = real_nap[6]+sim_nap[3]
        self.id7 = real_nap[7]+sim_nap[4]
        self.id8 = real_nap[8]-sim_nap[5]
    
    
    def map_sim_to_real(self, sim_positions = None):
        if self.id1 is None:
            print("Mapping is not set!!")
            exit(1)
        
        real_joints = {
            1: np.rad2deg(self.id1 + sim_positions[0]),
            2: np.rad2deg(self.id2 - sim_positions[1]),
            3: np.rad2deg(self.id3 + sim_positions[1]),
            4: np.rad2deg(self.id4 - sim_positions[2]),
            5: np.rad2deg(self.id5 + sim_positions[2]),
            6: np.rad2deg(self.id6 - sim_positions[3]),
            7: np.rad2deg(self.id7 - sim_positions[4]),
            8: np.rad2deg(self.id8 + sim_positions[5])
        }
        return real_joints
    
    def map_to_angles(self, sim_positions = None):
        
        real_joints = {
            1: np.rad2deg(sim_positions[0]),
            2: np.rad2deg(sim_positions[1]),
            3: np.rad2deg(sim_positions[1]),
            4: np.rad2deg(sim_positions[2]),
            5: np.rad2deg(sim_positions[2]),
            6: np.rad2deg(sim_positions[3]),
            7: np.rad2deg(sim_positions[4]),
            8: np.rad2deg(sim_positions[5])
        }
        return real_joints
    
    def map_adjustments(self):
        
        adjustmets = {
            1: np.rad2deg(self.id1),
            2: np.rad2deg(self.id2),
            3: np.rad2deg(self.id3),
            4: np.rad2deg(self.id4),
            5: np.rad2deg(self.id5),
            6: np.rad2deg(self.id6),
            7: np.rad2deg(self.id7),
            8: np.rad2deg(self.id8)
        }
        return adjustmets

    def set_position_from_sim(self, sim_positions):
        # print("Sim : ", self.map_to_angles(sim_positions))
        # print("Real: ", self.map_sim_to_real(sim_positions) )
        # print("Adju: ", self.map_adjustments() )
        self.set_position(self.map_sim_to_real(sim_positions))