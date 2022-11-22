'''
Written by Mr. Guy Tordjman 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ Dec 2021

Physical simulation is based the MuJoCo open source simulator (http://www.mujoco.org)
This class contains the kinematic function needed to calculate the arms configuration. 

'''


from re import T
import numpy as np
import sympy as sp
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
from sympy.matrices.dense import rot_axis1, rot_axis2, rot_axis3
class Kinematics:
    def __init__(self, robot):
        self.robot = robot

    
    def treat_as_zero(self, x):
        return abs(x)< 1e-6

    def w_to_skew(self, w):
        return np.array([[  0  ,-w[2],  w[1]],
                         [ w[2],  0  , -w[0]],
                         [-w[1], w[0],   0 ]])
    def skew_to_w(self, skew):
        return np.array([skew[2][1], skew[0][2], skew[1][0]])

    # using the rodriguez formula
    def skew_to_rot_mat(self, skew):
        w = self.skew_to_w(skew)
        theta = np.linalg.norm(w)
        if self.treat_as_zero(theta):
            return np.eye(3)
        skew_normed = skew/theta
        return np.eye(3)+np.sin(theta)*skew_normed + (1 - np.cos(theta)) * np.dot(skew_normed, skew_normed)

    def w_to_rot_mat(self, w):
        skew = self.w_to_skew(w)
        return self.skew_to_rot_mat(skew)
    

    # represents a 1*6 velocity vector v in  a 4x4 matrix for e^[v]
    def v6_to_matrix_expo_form(self, V):
        skew = self.w_to_skew([V[0], V[1], V[2]])
        v = [V[3], V[4], V[5]]
        skew_v = np.c_[skew, v]
        return np.r_[skew_v, np.zeros((1, 4))]
    
    # convert a matrix expo [V] ([S] or [B]) to its 6x1 vector representation
    def mat_exp_to_v(self, v_mat):
        return np.c_[v_mat[2][1], v_mat[0][2], v_mat[1][0]], [v_mat[0][3], v_mat[1][3], v_mat[2][3]]
    
    # converts a 4x4 se3 mat_exp [V] to a SE3 homogenious transformation matrix (htm)
    def mat_exp_to_htm(self, exp_mat):
        skew = exp_mat[0: 3, 0: 3]
        w = self.skew_to_w(skew)
        v = exp_mat[0: 3, 3]
        theta = np.linalg.norm(w)
        if self.treat_as_zero(theta):
            return np.r_[np.c_[np.eye(3), v], [[0, 0, 0, 1]]]
        
        skew_normed = skew / theta
        rotation_mat = self.w_to_rot_mat(w)
        g = np.eye(3) * theta + (1 - np.cos(theta)) * skew_normed + (theta - np.sin(theta)) * np.dot(skew_normed,skew_normed)
        cols = np.c_[rotation_mat, np.dot(g,v)/theta]
        return np.r_[cols,
                     [[0, 0, 0, 1]]]
    def v6_to_htm(self, v):
        exp_mat = self.v6_to_matrix_expo_form(v)
        return self.mat_exp_to_htm(exp_mat)
    
    def cross(self, w, q):
        a = w[1]*q[2] - w[2]*q[1]
        b = w[0]*q[2] - w[2]*q[0]
        c = w[0]*q[1] - w[1]*q[0]    
        r = np.array([a, -b, c])
        return r
   
        
    def translate(self):
        
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        z = sp.Symbol("z")

        trans = Matrix([            [1, 0, 0, x],
                                    [0, 1, 0, y],
                                    [0, 0, 1, z], 
                                    [0, 0, 0, 1]])
        return trans

    def htm_to_rp(self, T):
        T = np.array(T)
        return T[0: 3, 0: 3], T[0: 3, 3]

    def htm_inv(self, T):
        R, p = self.htm_to_rp(T)
        Rt = np.array(R).T
        return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]

    def htm_to_exp_mat(self, T):
        R, p = self.htm_to_rp(T)
        skew = self.rot_to_skew(R)
        p = [T[0][3], T[1][3], T[2][3]]
        if np.array_equal(skew, np.zeros((3, 3))):
            return np.r_[   np.c_[   np.zeros((3, 3)), p],
                            [[0, 0, 0, 0]]]
        else:
            theta = np.arccos((np.trace(R) - 1) / 2.0)
            g_inv = np.eye(3) - skew / 2.0 + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2)* np.dot(skew,skew) / theta
            p = [T[0][3],T[1][3],T[2][3]]
            v = np.dot(g_inv,p)
            return np.r_[np.c_[skew,v], [[0, 0, 0, 0]]]

     #Note that the omega vector is not normalized by theta as in the books algorithm                   
    def rot_to_skew(self, R):

        acosinput = (np.trace(R) - 1) / 2.0
        #pure translation
        if acosinput >= 1:
            return np.zeros((3, 3))
        elif acosinput <= -1:
            if not self.treat_as_zero(1 + R[2][2]):
                w = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                    * np.array([R[0][2], R[1][2], 1 + R[2][2]])
            elif not self.treat_as_zero(1 + R[1][1]):
                w = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                    * np.array([R[0][1], 1 + R[1][1], R[2][1]])
            else:
                w = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                    * np.array([1 + R[0][0], R[1][0], R[2][0]])
            return self.w_to_skew(np.pi * w)
        else:
            theta = np.arccos(acosinput)
            return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)
        
    
    def FK(self, M, s_poe, thetas):
        T = np.array(M)
        for i in range(len(thetas) - 1, -1, -1):
            mat_exp = self.v6_to_matrix_expo_form(np.array(s_poe)[:, i] * thetas[i])
            T = np.dot(self.mat_exp_to_htm(mat_exp), T)
        return T
    
    def Adj(self, T):
        R, p = self.htm_to_rp(T)
        return np.r_[np.c_[R, np.zeros((3, 3))],np.c_[np.dot(self.w_to_skew(p), R), R]]

    # devides the path to sections
    def trajectory(self, Tstart, Tend, sections):
        Tdif = Tend - Tstart
        traj_list = []
        for i in range(1, sections+1):
            traj_list.append(Tstart+(i/sections)*Tdif)
        return traj_list
    
    
    def CartesianTrajectory(self, Tend, Tf, N, method):
        t0 = self.robot.get_joints_pos()
        M = self.robot.M
        s_poe = self.robot.s_poe
        Tstart = self.FK(M, s_poe, t0)
        N = int(N)
        timegap = Tf / (N - 1.0)
        traj = [[None]] * N
        Rstart, pstart = self.htm_to_rp(Tstart)
        Rend, pend = self.htm_to_rp(Tend)
        for i in range(N):
            if method == 3:
                s = self.CubicTimeScaling(Tf, timegap * i)
            else:
                s = self.QuinticTimeScaling(Tf, timegap * i)
            traj[i] \
            = np.r_[np.c_[np.dot(Rstart, \
            self.skew_to_rot_mat(self.rot_to_skew(np.dot(np.array(Rstart).T,Rend)) * s)), \
                    s * np.array(pend) + (1 - s) * np.array(pstart)], \
                    [[0, 0, 0, 1]]]
        return traj
    def CubicTimeScaling(self, Tf, t):
        """Computes s(t) for a cubic time scaling
        :param Tf: Total time of the motion in seconds from rest to rest
        :param t: The current time t satisfying 0 < t < Tf
        :return: The path parameter s(t) corresponding to a third-order
                polynomial motion that begins and ends at zero velocity
        Example Input:
            Tf = 2
            t = 0.6
        Output:
            0.216
        """
        return 3 * (1.0 * t / Tf) ** 2 - 2 * (1.0 * t / Tf) ** 3

    def QuinticTimeScaling(self, Tf, t):
        """Computes s(t) for a quintic time scaling
        :param Tf: Total time of the motion in seconds from rest to rest
        :param t: The current time t satisfying 0 < t < Tf
        :return: The path parameter s(t) corresponding to a fifth-order
                polynomial motion that begins and ends at zero velocity and zero
                acceleration
        Example Input:
            Tf = 2
            t = 0.6
        Output:
            0.16308
        """
        return 10 * (1.0 * t / Tf) ** 3 - 15 * (1.0 * t / Tf) ** 4 \
            + 6 * (1.0 * t / Tf) ** 5
    
    # when using the newton raphson method a guess theta value is required as a starting point to find a local minima.
    # this sometimes results in failure to find a solution
    # if the desired EE configuration is close to the guess the method converges fastly.
    # by setting the guess to the current position and the desired EE conf. to a configuration that is closer to the starting point 
    # we improve the accuarcy of our result and guarantee to find a local minima.
    # 
    # this method iterates through a list of ordered configuration (from starting to desired configuration), each iteration the IK 
    # is calculated. At the last iteration the thetas are returned from the IK of the final configuration.

    def trajectoryIK(self,T_target, T_start,  w_err, v_err, sections):
        #current EE conf is the starting configuration
        t0 = self.robot.get_joints_pos()
        M = self.robot.M
        s_poe = self.robot.s_poe
        if T_start is None:
            Tstart = self.FK(M, s_poe, t0)

        #get list of ordered transformation matrices from current EE config to desired one
        traj_list = self.trajectory(Tstart, T_target, sections)
        #calculate IK for each section. set the guess for the following section from the IK of the previous section
        for i in range(1, len(traj_list)):
            next_T = traj_list[i]
            prev_t0 = t0.copy()
            t0 = self.IK_space(s_poe, M, next_T, prev_t0, w_err, v_err)[0]
           
        return t0
     
    def IK(self, T_target, w_err, v_err):
        #current EE conf is the starting configuration
        t0 = self.robot.get_joints_pos()
        M = self.robot.M
        s_poe = self.robot.s_poe
        return self.IK_space(s_poe, M, T_target, t0, w_err, v_err)[0]
    
    def IK_space(self, s_poe, M, Tsd, t0, w_err, v_err, max_iter=20):
        
        i = 0
        # guess input
        thetas = np.array(t0).copy()
        
        #calculating current Tbd by finding current Tsb^-1 (EE configuration inversed)  and multiply it with 
        # a given desired EE configuration Tsd 
        Tsb = self.FK(M,s_poe, thetas)
        Tbs = self.htm_inv(Tsb)
        Tbd = np.dot(Tbs, Tsd)
        # respresenting Tbd in matrix exponential form 
        Tbd_mat_exp = self.htm_to_exp_mat(Tbd)
        # twist of the body in respect to the desired EE
        Vb = self.mat_exp_to_v(Tbd_mat_exp)
        # twist of the s frame in respect to the body
        Vs = np.dot(self.Adj(Tsb),Vb)
        # check if a solution is yet to be found
        keep_looking = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > w_err \
            or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > v_err
        
        while keep_looking and  i < max_iter:
            #get current jacobian and calculate its pseudo inverse
            j_s = self.JacobianSpace(thetas)
            j_s_pinv = np.linalg.pinv(j_s)
            
            thetas = thetas + np.dot(j_s_pinv, Vs)
            i = i + 1
            Tsb = self.FK(M,s_poe, thetas)
            Tbs = self.htm_inv(Tsb)
            Tbd = np.dot(Tbs, Tsd)
            Tbd_mat_exp = self.htm_to_exp_mat(Tbd)
            Vs = np.dot(self.Adj(Tsb),self.mat_exp_to_v(Tbd_mat_exp))

            keep_looking = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > w_err \
                or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > v_err
        
        return (thetas, not keep_looking)


    def mat_exp_to_v(self, mat_exp):
        return np.r_[[mat_exp[2][1], mat_exp[0][2], mat_exp[1][0]],
                    [mat_exp[0][3], mat_exp[1][3], mat_exp[2][3]]]

    def JacobianSpace(self, thetas):
        # start with the spacial screw axis and Identity mat.
        T_0i = np.eye(4)
        Js = np.array(self.robot.s_poe).copy().astype(np.float64)

        for i in range(1, len(thetas)):
            # calculate Ti-1,i from its matrix exponential form
            prev_S = self.robot.s_poe[:, i - 1]
            prev_S_theta = prev_S * thetas[i - 1]
            prev_S_theta_mat_exp = self.v6_to_matrix_expo_form(prev_S_theta)
            T_i = self.mat_exp_to_htm(prev_S_theta_mat_exp)
            # calculate T_0,i by multiplying T[0,i] = T[0,i-1] * T[i-1,i]  
            T_0i = np.dot(T_0i, T_i)
            # use the adjoint to calulate the jacoban column as in Jsi = Adj(T_0i)*Si (equation 5.11)
            Js[:, i] = np.dot(self.Adj(T_0i), np.array(self.robot.s_poe)[:, i])
        return Js

