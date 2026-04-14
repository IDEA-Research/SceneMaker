import numpy as np
import os
import json
import cv2
import trimesh
import pickle as p
import argparse
from tqdm import tqdm

import pdb

''' common world coordinate system conventions

   OpenGL        OpenCV Colmap       Blender        Unity             
Right-handed      Left-handed     Right-handed   Left-handed  

     +y                +z           +z  +y         +y  +z                                               
     |                /             |  /           |  /                                               
     |               /              | /            | /                                                   
     |______+x      /______+x       |/_____+x      |/_____+x                                          
    /               |                                                                                        
   /                |                                                                                                  
  /                 |                                                                                         
 +z                 +y                                                                                           

'''
def convert_camera_coordinate(
    pose, 
    target: str = 'opengl', 
    original: str = 'blender',
):
    """A method to convert between different world coordinate systems.

    Args:
    - pose (np.ndarray): camera pose, float [4, 4].
    - target (Literal[&#39;unity&#39;, &#39;blender&#39;, &#39;opencv&#39;, &#39;colmap&#39;, &#39;opengl&#39;], optional): from convention. Defaults to 'unity'.
    - original (Literal[&#39;unity&#39;, &#39;blender&#39;, &#39;opencv&#39;, &#39;colmap&#39;, &#39;opengl&#39;], optional): to convention. Defaults to 'opengl'.

    Returns:
    - np.ndarray: converted camera pose, float [4, 4].
    """
    
    if original == 'opengl':
        if target == 'unity':
            pose[2] *= -1
        elif target == 'blender':
            pose[2] *= -1
            pose[[1, 2]] = pose[[2, 1]]
        elif target in ['opencv', 'colmap']:
            pose[1:3] *= -1
    elif original == 'unity':
        if target == 'opengl':
            pose[2] *= -1
        elif target == 'blender':
            pose[[1, 2]] = pose[[2, 1]]
        elif target in ['opencv', 'colmap']:
            pose[1] *= -1
    elif original == 'blender':
        if target == 'opengl':
            pose[1] *= -1
            pose[[1, 2]] = pose[[2, 1]]
        elif target == 'unity':
            pose[[1, 2]] = pose[[2, 1]]
        elif target in ['opencv', 'colmap']:
            pose[2] *= -1
            pose[[1, 2]] = pose[[2, 1]]
    elif original in ['opencv', 'colmap']:
        if target == 'opengl':
            pose[1:3] *= -1
        elif target == 'unity':
            pose[1] *= -1
        elif target == 'blender':
            pose[1] *= -1
            pose[[1, 2]] = pose[[2, 1]]
    return pose


def q2rot(Q):
    # Extract the values from Q
    x = Q[0]
    y = Q[1]
    z = Q[2]
    w = Q[3]
    x2=x+x
    y2=y+y
    z2=z+z
    xx=x*x2
    xy=x*y2
    xz=x*z2
    yy=y*y2
    yz=y*z2
    zz=z*z2
    wx=w*x2
    wy=w*y2
    wz=w*z2
    # First row of the rotation matrix
    r00 = (1-(yy+zz))
    r01 = xy+wz
    r02 = xz-wy
    # Second row of the rotation matrix
    r10 = xy-wz
    r11 = (1-(xx+zz))
    r12 = yz+wx
    # Third row of the rotation matrix
    r20 = xz+wy
    r21 = yz-wx
    r22 = 1-(xx+yy)

    # 3x3 rotation matrix
    pos = np.array([[r00, r10, r20],
                    [r01, r11, r21],
                    [r02, r12, r22],
                    ])
    return pos

def reconstruct_pcd(depth, intrinsic, cam2wrd_rot, scale=1, normalize=False, valid_mask=None):
    
    # apply scale on intrinsic
    intrinsic = intrinsic.copy()
    intrinsic[0, 0] *= scale
    intrinsic[1, 1] *= scale
    intrinsic[0, 2] *= scale
    intrinsic[1, 2] *= scale
    
    if valid_mask is not None:
        valid_Y, valid_X = np.where((depth > 0) & (depth < 10) & valid_mask)
    else:
        valid_Y, valid_X = np.where((depth > 0) & (depth < 10))

    max_num_points = 100000
    if valid_Y.shape[0] > max_num_points:
        random_ind = np.random.choice(valid_Y.shape[0], max_num_points, replace=False)
        valid_Y = valid_Y[random_ind]
        valid_X = valid_X[random_ind]

    unprojected_X = valid_X
    unprojected_Y = valid_Y
    unprojected_Z = np.ones_like(unprojected_X)
    point_cloud_xyz = np.stack([unprojected_X, unprojected_Y, unprojected_Z], axis=1)

    intrinsic_inv = np.linalg.inv(intrinsic[0:3, 0:3])
    point_cloud_incam = (intrinsic_inv @ point_cloud_xyz.T).T * depth[valid_Y, valid_X][:,None]
    point_cloud_incam = -point_cloud_incam[:, :3].copy()
    # point_cloud_incam[:,:2] = -point_cloud_incam[:,:2].copy() # raw data

    if normalize:
        translation = (np.max(point_cloud_incam, axis=0) + np.min(point_cloud_incam, axis=0)) / 2
        point_cloud_incam = point_cloud_incam - translation
        scale = np.max(np.abs(point_cloud_incam[:, :2]))
        point_cloud_incam = point_cloud_incam / scale
        return point_cloud_incam, translation, scale
    else:
        return point_cloud_incam
    

def get_pitch_from_R(cam_R):
    angle=np.arctan2(cam_R[1,2],cam_R[1,1])
    roll=np.arctan2(-cam_R[1,0],cam_R[0,0])
    return angle

def get_rot_from_pitch(pitch):
    cp=np.cos(pitch)
    sp=np.sin(pitch)
    rot=np.array([[1,0,0],
              [0,cp,-sp],
              [0,sp,cp]])
    return rot

def get_rot_from_yaw(yaw):
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    rot = np.array([[ cy, 0, sy],
                    [  0, 1,  0],
                    [-sy, 0, cy]])
    return rot

def get_yaw_from_R(cam_R):
    # cam_R: 3x3 rotation matrix
    # yaw = atan2(cam_R[0,2], cam_R[0,0])
    yaw = np.arctan2(cam_R[0,2], cam_R[0,0])
    return yaw