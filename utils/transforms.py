import numpy as np
import torch

############## angle-axis representation ################
def aa2quat(rots, form='wxyz', unified_orient=True):
    """
    Converts angle-axis representation to wxyz quaternion and to the half plan (w >= 0)
    @param rots: angle-axis rotations, (*, 3)
    @param form: quaternion format, either 'wxyz' or 'xyzw'
    @param unified_orient: Use unified orientation for quaternion (quaternion is dual cover of SO3)
    :return: wxyz quaternion
    """
    is_numpy = False
    if isinstance(rots, np.ndarray):
        rots = torch.from_numpy(rots)
        is_numpy = True
    angles = rots.norm(dim=-1, keepdim=True)
    norm = angles.clone()
    norm[norm < 1e-8] = 1
    axis = rots / norm
    quats = torch.empty(rots.shape[:-1] + (4,), device=rots.device, dtype=rots.dtype)
    angles = angles * 0.5
    if form == 'wxyz':
        quats[..., 0] = torch.cos(angles.squeeze(-1))
        quats[..., 1:] = torch.sin(angles) * axis
    elif form == 'xyzw':
        quats[..., :3] = torch.sin(angles) * axis
        quats[..., 3] = torch.cos(angles.squeeze(-1))

    if unified_orient:
        idx = quats[..., 0] < 0
        quats[idx, :] *= -1

    return quats.numpy() if is_numpy else quats


def aa2mat(rots):
    """
    Converts angle-axis representation to rotation matrix
    :param rots: angle-axis representation
    :return: rotation matrix
    """
    is_numpy = False
    if isinstance(rots, np.ndarray):
        rots = torch.from_numpy(rots)
        is_numpy = True
    quat = aa2quat(rots)
    mat = quat2mat(quat)
    return mat.numpy() if is_numpy else mat


def aa2euler(rots, order='xyz', degrees=True):
    """
    Converts angle-axis representation to xyz euler angles
    :param rots: angle-axis representation
    :param order: euler angle order
    :param degrees: whether to return degrees
    :return: xyz euler angles
    """
    is_numpy = False
    if isinstance(rots, np.ndarray):
        rots = torch.from_numpy(rots)
        is_numpy = True
    quat = aa2quat(rots)
    euler = quat2euler(quat, order, degrees)
    return euler.numpy() if is_numpy else euler


def aa2repr6d(rots):
    """
    Converts angle-axis representation to 6D representation
    :param rots: angle-axis representation
    :return: 6D representation
    """
    is_numpy = False
    if isinstance(rots, np.ndarray):
        rots = torch.from_numpy(rots)
        is_numpy = True
    quat = aa2quat(rots)
    repr6d = quat2repr6d(quat)
    return repr6d.numpy() if is_numpy else repr6d


def aa2rodrigues(rots):
    """
    Converts angle-axis representation to Rodrigues vector
    :param rots: angle-axis representation
    :return: Rodrigues vector
    """
    is_numpy = False
    if isinstance(rots, np.ndarray):
        rots = torch.from_numpy(rots)
        is_numpy = True
    quat = aa2quat(rots)
    rot_vec = quat2rodrigues(quat)
    return rot_vec.numpy() if is_numpy else rot_vec


############## quaternion representation ################
def quat2aa(quats):
    """
    Converts wxyz quaternions to angle-axis representation
    :param quats: wxyz quaternions
    :return: angle-axis representation
    """
    is_numpy = False
    if isinstance(quats, np.ndarray):
        quats = torch.from_numpy(quats)
        is_numpy = True
    _cos = quats[..., 0]
    xyz = quats[..., 1:]
    _sin = xyz.norm(dim=-1)
    norm = _sin.clone()
    norm[norm < 1e-7] = 1
    axis = xyz / norm.unsqueeze(-1)
    angle = axis * torch.atan2(_sin, _cos).unsqueeze(-1) * 2
    return angle.numpy() if is_numpy else angle


def quat2mat(quats):
    """
    Converts (w, x, y, z) quaternions to 3x3 rotation matrix
    :param quats: quaternions of shape (..., 4)
    :return: rotation matrices of shape (..., 3, 3)
    """
    is_numpy = False
    if isinstance(quats, np.ndarray):
        quats = torch.from_numpy(quats)
        is_numpy = True

    norm_quat = quats
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    batch_size, _ = quats.shape
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    mat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(batch_size, 3, 3)
    return mat.numpy() if is_numpy else mat


def quat2euler(q, order='xyz', degrees=True):
    """
    Converts (w, x, y, z) quaternions to euler angles
    :param q: quaternions of shape (..., 4)
    :param order: euler angle order
    :param degrees: whether to return degrees
    :return: euler angles of shape (..., 3)
    """
    is_numpy = False
    if isinstance(q, np.ndarray):
        q = torch.from_numpy(q)
        is_numpy = True
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]
    es = torch.empty(q0.shape + (3,), device=q.device, dtype=q.dtype)

    if order == 'xyz':
        es[..., 2] = torch.atan2(2 * (q0 * q3 - q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
        es[..., 1] = torch.asin((2 * (q1 * q3 + q0 * q2)).clip(-1, 1))
        es[..., 0] = torch.atan2(2 * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
    else:
        raise NotImplementedError('Cannot convert to ordering %s' % order)

    if degrees:
        es = es * 180 / np.pi

    return es.numpy() if is_numpy else es


def quat2repr6d(quat):
    """
    Converts wxyz quaternions to 6D representation
    :param quat: wxyz quaternions
    :return: 6D representation
    """
    is_numpy = False
    if isinstance(quat, np.ndarray):
        quat = torch.from_numpy(quat)
        is_numpy = True
    mat = quat2mat(quat)
    res = mat[..., :2, :]
    res = res.reshape(res.shape[:-2] + (6, ))
    return res.numpy() if is_numpy else res


def quat2rodrigues(quat):
    """
    Converts wxyz quaternions to Rodrigues vector
    :param quat: wxyz quaternions
    :return: Rodrigues vector
    """
    is_numpy = False
    if isinstance(quat, np.ndarray):
        quat = torch.from_numpy(quat)
        is_numpy = True
    mat = quat2mat(quat)
    rot_vec = mat2rodrigues(mat)
    return rot_vec.numpy() if is_numpy else rot_vec

############## euler representation ################
def euler2aa(rots, order='xyz', degrees=True):
    """
    Converts xyz euler angles to angle-axis representation
    :param rots: xyz euler angles
    :param order: euler angle order
    :param degrees: whether to return degrees
    :return: angle-axis representation
    """
    is_numpy = False
    if isinstance(rots, np.ndarray):
        rots = torch.from_numpy(rots)
        is_numpy = True
    if degrees:
        rots = rots / 180 * np.pi
    mats = euler2mat(rots, order)
    aa = mat2aa(mats)
    return aa.numpy() if is_numpy else aa


def euler2mat(rots, order='xyz', degrees=True):
    """
    Converts xyz euler angles to rotation matrix
    :param rots: xyz euler angles
    :param order: euler angle order
    :return: rotation matrix
    """
    axis = {'x': torch.tensor((1, 0, 0), device=rots.device),
            'y': torch.tensor((0, 1, 0), device=rots.device),
            'z': torch.tensor((0, 0, 1), device=rots.device)}

    is_numpy = False
    if isinstance(rots, np.ndarray):
        rots = torch.from_numpy(rots)
        is_numpy = True
    if degrees:
        rots = rots / 180 * np.pi
    mats = []
    for i in range(3):
        aa = axis[order[i]] * rots[..., i].unsqueeze(-1)
        mats.append(aa2mat(aa))
    return (mats[0] @ (mats[1] @ mats[2])).numpy() if is_numpy else (mats[0] @ (mats[1] @ mats[2]))


def euler2quat(rots, order='xyz', degrees=True):
    """
    Converts xyz euler angles to wxyz quaternion
    :param rots: xyz euler angles
    :param order: euler angle order
    :param degrees: whether to return degrees
    :return: wxyz quaternion
    """
    is_numpy = False
    if isinstance(rots, np.ndarray):
        rots = torch.from_numpy(rots)
        is_numpy = True
    if degrees:
        rots = rots / 180 * np.pi
    mats = euler2mat(rots, order)
    quat = mat2quat(mats)
    return quat.numpy() if is_numpy else quat


def euler2repr6d(rots, order='xyz', degrees=True):
    """
    Converts xyz euler angles to 6D representation
    :param rots: xyz euler angles
    :param order: euler angle order
    :param degrees: whether to return degrees
    :return: 6D representation
    """
    is_numpy = False
    if isinstance(rots, np.ndarray):
        rots = torch.from_numpy(rots)
        is_numpy = True
    if degrees:
        rots = rots / 180 * np.pi
    mats = euler2mat(rots, order)
    repr6d = mat2repr6d(mats)
    return repr6d.numpy() if is_numpy else repr6d


def euler2rodrigues(rots, order='xyz', degrees=True):
    """
    Converts xyz euler angles to Rodrigues vector
    :param rots: xyz euler angles
    :param order: euler angle order
    :param degrees: whether to return degrees
    :return: Rodrigues vector
    """
    is_numpy = False
    if isinstance(rots, np.ndarray):
        rots = torch.from_numpy(rots)
        is_numpy = True
    if degrees:
        rots = rots / 180 * np.pi
    mats = euler2mat(rots, order)
    rot_vec = mat2rodrigues(mats)
    return rot_vec.numpy() if is_numpy else rot_vec


############## rotation matrix representation ################
def mat2aa(R):
    """
    Converts rotation matrix to angle-axis representation
    :param R: rotation matrix
    :return: angle-axis representation
    """
    is_numpy = False
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R)
        is_numpy = True
    quat = mat2quat(R)
    aa = quat2aa(quat)
    return aa.numpy() if is_numpy else aa


def mat2quat(R):
    '''
    Converts a rotation matrix to a unit quaternion.

    Args:
    - R: rotation matrix tensor of shape (..., 3, 3)

    Returns:
    - q: unit quaternion tensor of shape (..., 4)

    Reference:
    - https://github.com/duolu/pyrotation/blob/master/pyrotation/pyrotation.py
    - Shepperd’s method for numerical stability is used.

    Note:
    - The rotation matrix must be orthonormal.
    '''
    is_numpy = False
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R)
        is_numpy = True
    w2 = (1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2])
    x2 = (1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2])
    y2 = (1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2])
    z2 = (1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2])

    yz = (R[..., 1, 2] + R[..., 2, 1])
    xz = (R[..., 2, 0] + R[..., 0, 2])
    xy = (R[..., 0, 1] + R[..., 1, 0])

    wx = (R[..., 2, 1] - R[..., 1, 2])
    wy = (R[..., 0, 2] - R[..., 2, 0])
    wz = (R[..., 1, 0] - R[..., 0, 1])

    w = torch.empty_like(x2)
    x = torch.empty_like(x2)
    y = torch.empty_like(x2)
    z = torch.empty_like(x2)

    flagA = (R[..., 2, 2] < 0) * (R[..., 0, 0] > R[..., 1, 1])
    flagB = (R[..., 2, 2] < 0) * (R[..., 0, 0] <= R[..., 1, 1])
    flagC = (R[..., 2, 2] >= 0) * (R[..., 0, 0] < -R[..., 1, 1])
    flagD = (R[..., 2, 2] >= 0) * (R[..., 0, 0] >= -R[..., 1, 1])

    x[flagA] = torch.sqrt(x2[flagA])
    w[flagA] = wx[flagA] / x[flagA]
    y[flagA] = xy[flagA] / x[flagA]
    z[flagA] = xz[flagA] / x[flagA]

    y[flagB] = torch.sqrt(y2[flagB])
    w[flagB] = wy[flagB] / y[flagB]
    x[flagB] = xy[flagB] / y[flagB]
    z[flagB] = yz[flagB] / y[flagB]

    z[flagC] = torch.sqrt(z2[flagC])
    w[flagC] = wz[flagC] / z[flagC]
    x[flagC] = xz[flagC] / z[flagC]
    y[flagC] = yz[flagC] / z[flagC]

    w[flagD] = torch.sqrt(w2[flagD])
    x[flagD] = wx[flagD] / w[flagD]
    y[flagD] = wy[flagD] / w[flagD]
    z[flagD] = wz[flagD] / w[flagD]

    res = [w, x, y, z]
    res = [z.unsqueeze(-1) for z in res]

    return torch.cat(res, dim=-1).numpy() if is_numpy else torch.cat(res, dim=-1)


def mat2euler(R, order='xyz', degrees=True):
    """
    Converts rotation matrix to xyz euler angles
    :param R: rotation matrix
    :param order: euler angle order
    :param degrees: whether to return degrees
    :return: xyz euler angles
    """
    is_numpy = False
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R)
        is_numpy = True
    quat = mat2quat(R)
    euler = quat2euler(quat, order, degrees)
    return euler.numpy() if is_numpy else euler


def mat2repr6d(R):
    """
    Converts rotation matrix to 6D representation
    :param R: rotation matrix
    :return: 6D representation
    """
    is_numpy = False
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R)
        is_numpy = True
    quat = mat2quat(R)
    repr6d = quat2repr6d(quat)
    return repr6d.numpy() if is_numpy else repr6d


def mat2rodrigues(R):
    """
    Converts rotation matrix to Rodrigues vector
    :param R: rotation matrix
    :return: Rodrigues vector
    """
    is_numpy = False
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R)
        is_numpy = True

    theta = torch.acos((torch.einsum("...ii", R) - 1) / 2)
    batch_size, _, _ = R.shape

    r = torch.zeros(batch_size, 3)
    r[..., 0] = (R[..., 2, 1] - R[..., 1, 2]) / (2 * torch.sin(theta))
    r[..., 1] = (R[..., 0, 2] - R[..., 2, 0]) / (2 * torch.sin(theta))
    r[..., 2] = (R[..., 1, 0] - R[..., 0, 1]) / (2 * torch.sin(theta))    
    
    rot_vec = theta.unsqueeze(-1) * r

    return rot_vec.numpy() if is_numpy else rot_vec


############## 6D representation ################
def repr6d2aa(repr):
    """
    Converts 6D representation to angle-axis representation
    :param repr: 6D representation
    :return: angle-axis representation
    """
    is_numpy = False
    if isinstance(repr, np.ndarray):
        repr = torch.from_numpy(repr)
        is_numpy = True
    mat = repr6d2mat(repr)
    aa = mat2aa(mat)
    return aa.numpy() if is_numpy else aa


def repr6d2quat(repr):
    """
    Converts 6D representation to wxyz quaternion
    :param repr: 6D representation
    :return: wxyz quaternion
    """
    is_numpy = False
    if isinstance(repr, np.ndarray):
        repr = torch.from_numpy(repr)
        is_numpy = True
    x = repr[..., :3]
    y = repr[..., 3:]
    x = x / x.norm(dim=-1, keepdim=True)
    z = torch.cross(x, y)
    z = z / z.norm(dim=-1, keepdim=True)
    y = torch.cross(z, x)
    res = [x, y, z]
    res = [v.unsqueeze(-2) for v in res]
    mat = torch.cat(res, dim=-2)
    return mat2quat(mat).numpy() if is_numpy else mat2quat(mat)


def repr6d2euler(repr, order='xyz', degrees=True):
    """
    Converts 6D representation to xyz euler angles
    :param repr: 6D representation
    :param order: euler angle order
    :param degrees: whether to return degrees
    :return: xyz euler angles
    """
    is_numpy = False
    if isinstance(repr, np.ndarray):
        repr = torch.from_numpy(repr)
        is_numpy = True
    mat = repr6d2mat(repr)
    euler = mat2euler(mat, order, degrees)
    return euler.numpy() if is_numpy else euler


def repr6d2mat(repr):
    """
    Converts 6D representation to rotation matrix
    :param repr: 6D representation
    :return: rotation matrix
    """
    is_numpy = False
    if isinstance(repr, np.ndarray):
        repr = torch.from_numpy(repr)
        is_numpy = True
    x = repr[..., :3]
    y = repr[..., 3:]
    x = x / x.norm(dim=-1, keepdim=True)
    z = torch.cross(x, y)
    z = z / z.norm(dim=-1, keepdim=True)
    y = torch.cross(z, x)
    res = [x, y, z]
    res = [v.unsqueeze(-2) for v in res]
    mat = torch.cat(res, dim=-2)
    return mat.numpy() if is_numpy else mat


def repr6d2rodrigues(repr):
    """
    Converts 6D representation to Rodrigues vector
    :param repr: 6D representation
    :return: Rodrigues vector
    """
    is_numpy = False
    if isinstance(repr, np.ndarray):
        repr = torch.from_numpy(repr)
        is_numpy = True
    mat = repr6d2mat(repr)
    rot_vec = mat2rodrigues(mat)
    return rot_vec.numpy() if is_numpy else rot_vec


############## Rodrigues representation ################
def rodrigues2aa(rot_vec):
    """
    Converts a Rodrigues vector to angle-axis representation
    :param rot_vec: a 3D vector representing the axis of rotation scaled by the angle of rotation
    :return: angle-axis representation
    """
    is_numpy = False
    if isinstance(rot_vec, np.ndarray):
        rot_vec = torch.from_numpy(rot_vec)
        is_numpy = True
    mat = rodrigues2mat(rot_vec)
    aa = mat2aa(mat)
    return aa.numpy() if is_numpy else aa


def rodrigues2mat(rot_vec):
    """
    Converts a Rodrigues vector to a rotation matrix
    :param rot_vec: a 3D vector representing the axis of rotation scaled by the angle of rotation
    :return: rotation matrix
    """
    is_numpy = False
    if isinstance(rot_vec, np.ndarray):
        rot_vec = torch.from_numpy(rot_vec)
        is_numpy = True
    angle = torch.linalg.norm(rot_vec, dim=-1, keepdim=True) # calculate the angle of rotation
    axis = rot_vec / angle # normalize the axis of rotation
    batch_size = axis.shape[0]
    skew_symmetric_matrix = torch.zeros((batch_size, 3, 3), dtype=torch.float64, device=rot_vec.device)
    skew_symmetric_matrix[:, 0, 1] = -axis[..., 2]
    skew_symmetric_matrix[:, 0, 2] = axis[..., 1]
    skew_symmetric_matrix[:, 1, 0] = axis[..., 2]
    skew_symmetric_matrix[:, 1, 2] = -axis[..., 0]
    skew_symmetric_matrix[:, 2, 0] = -axis[..., 1]
    skew_symmetric_matrix[:, 2, 1] = axis[..., 0]

    axis = axis.reshape(*axis.shape[:-1], 3, 1) # reshape the axis of rotation
    axis_times_axis_transpose = axis.matmul(axis.transpose(-1, -2)) # calculate the outer product of the axis of rotation

    angle = angle.unsqueeze(-1).repeat(1, 3, 3) # repeat the angle of rotation
    mat = torch.cos(angle) * torch.eye(3, dtype=torch.float64, device=rot_vec.device).unsqueeze(0).repeat(batch_size, 1, 1) + (1 - torch.cos(angle)) * axis_times_axis_transpose + torch.sin(angle) * skew_symmetric_matrix

    return mat.numpy() if is_numpy else mat


def rodrigues2quat(rot_vec):
    """
    Converts a Rodrigues vector to wxyz quaternion
    :param rot_vec: a 3D vector representing the axis of rotation scaled by the angle of rotation
    :return: wxyz quaternion
    """
    is_numpy = False
    if isinstance(rot_vec, np.ndarray):
        rot_vec = torch.from_numpy(rot_vec)
        is_numpy = True
    mat = rodrigues2mat(rot_vec)
    quat = mat2quat(mat)
    return quat.numpy() if is_numpy else quat


def rodrigues2euler(rot_vec, order='xyz', degrees=True):
    """
    Converts a Rodrigues vector to xyz euler angles
    :param rot_vec: a 3D vector representing the axis of rotation scaled by the angle of rotation
    :param order: euler angle order
    :param degrees: whether to return degrees
    :return: xyz euler angles
    """
    is_numpy = False
    if isinstance(rot_vec, np.ndarray):
        rot_vec = torch.from_numpy(rot_vec)
        is_numpy = True
    mat = rodrigues2mat(rot_vec)
    euler = mat2euler(mat, order, degrees)
    return euler.numpy() if is_numpy else euler


def rodriuges2repr6d(rot_vec):
    """
    Converts a Rodrigues vector to 6D representation
    :param rot_vec: a 3D vector representing the axis of rotation scaled by the angle of rotation
    :return: 6D representation
    """
    is_numpy = False
    if isinstance(rot_vec, np.ndarray):
        rot_vec = torch.from_numpy(rot_vec)
        is_numpy = True
    mat = rodrigues2mat(rot_vec)
    repr6d = mat2repr6d(mat)
    return repr6d.numpy() if is_numpy else repr6d


def recover_transform_matrix(rotation_6d, translation):
    """
    Recover 4x4 transformation matrix from 6D rotation representation and 3D translation.
    
    Args:
        rotation_6d: Tensor of shape (..., 6), representing 6D rotation.
        translation: Tensor of shape (..., 3), representing 3D translation.
    
    Returns:
        transformation: Tensor of shape (..., 4, 4)
    """
    
    R = repr6d2mat(rotation_6d)
    T = torch.eye(4, device=rotation_6d.device).expand(*rotation_6d.shape[:-1], 4, 4).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = translation
    
    return T