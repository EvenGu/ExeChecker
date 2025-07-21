import torch
import numpy as np
from utility.rotation import *
import random

#### used in training ###############################
def to_onehot(num_class, label, alpha=0):
    return torch.zeros((label.shape[0], num_class)).fill_(alpha).scatter_(1, label.unsqueeze(1), 1 - alpha)


def mixup(input, target, gamma):
    # target is onehot format!
    perm = torch.randperm(input.size(0))
    perm_input = input[perm]
    perm_target = target[perm]
    return input.mul_(gamma).add_(perm_input, alpha=1-gamma), target.mul_(gamma).add_(perm_target, alpha=1-gamma)


#### skeleton ######################################

def decouple_spatial(skeleton, edges=()): # CTVM
    assert skeleton.shape[-1] == 1 # M=1
    tmp = np.zeros(skeleton.shape)
    for v1, v2 in edges:
        tmp[:, :, v2] = skeleton[:, :, v2] - skeleton[:, :, v1]
    return tmp


def decouple_temporal(skeleton, inter_frame=1):  # CTVM
    assert skeleton.shape[-1] == 1 # M=1
    skeleton = skeleton[:, ::inter_frame]
    diff = skeleton[:, 1:] - skeleton[:, :-1]
    return diff


def calculate_orientation(skeleton, plevis_id=0, hip_left_id=10, neck_id=3): # CTVM
    '''
    re-define a coordinates system that is relative to pelvis in the first frame
    default values from the execheck joint order
    @param: 
        plevis_id: root joint
        hip_left_id: use pelvis-to-hip_left as x-asis, default hip_left = 10
        neck_id: use pelvis-to-neck as y-asis, default neck = 3
    @returns:
        joint's orientation in the re-defined coordinates system
    '''
    C, T, V, M = skeleton.shape
    assert M == 1
    skeleton = skeleton.squeeze()
    pelvis_pos = skeleton[:, 0, plevis_id] # root: pelvis position in the first frame
    # Compute the vectors for the y and x axes
    pelvis_to_neck = skeleton[:, 0, neck_id] - pelvis_pos
    pelvis_to_hip_left = skeleton[:, 0, hip_left_id] - pelvis_pos
    # Normalize the vectors to get unit vectors
    y_axis = pelvis_to_neck / np.linalg.norm(pelvis_to_neck)
    x_axis = pelvis_to_hip_left / np.linalg.norm(pelvis_to_hip_left)
    # Ensure y_axis and x_axis are orthogonal by recomputing x_axis
    z_axis = np.cross(y_axis, x_axis)
    x_axis = np.cross(z_axis, y_axis)

    # Compute the rotation matrix to align the pelvis frame to the global frame
    R_pelvis = np.vstack([x_axis, y_axis, z_axis]).T # (3,3)

    # Compute the relative orientations of all joints
    relative_positions = skeleton - pelvis_pos[:, np.newaxis, np.newaxis] # (3,T,V)
    relative_orientations = np.einsum('ij, jkl -> ikl', R_pelvis, relative_positions)

    relative_orientations[:, 0, 0] = pelvis_pos # keep the pelvis position in the first frame
    return relative_orientations[..., np.newaxis]    


#### not used ### 

def calculate_RPY(skeleton, plevis_id=0, hip_left_id=10, neck_id=3):
    pass


def rot_to_fix_angle_fstframe(skeleton, jpts=[0, 1], axis=[0, 0, 1], frame=0, person=0):
    '''
    :param skeleton: c t v m
    :param axis: 001 for z, 100 for x, 010 for y
    '''
    skeleton = np.transpose(skeleton, [3, 1, 2, 0])  # M, T, V, C
    joint_bottom = skeleton[person, frame, jpts[0]]
    joint_top = skeleton[person, frame, jpts[1]]
    axis_c = np.cross(joint_top - joint_bottom, axis)
    angle = angle_between(joint_top - joint_bottom, axis)
    matrix_z = rotation_matrix(axis_c, angle)
    tmp = np.dot(np.reshape(skeleton, (-1, 3)), matrix_z.transpose())
    skeleton = np.reshape(tmp, skeleton.shape)
    return skeleton.transpose((3, 1, 2, 0))


def sub_center_jpt_fstframe(skeleton, jpt=0, frame=0, person=0):
    C, T, V, M = skeleton.shape
    skeleton = np.transpose(skeleton, [3, 1, 2, 0])  # M, T, V, C
    main_body_center = skeleton[person, frame, jpt].copy()  # c
    for i_p, person in enumerate(skeleton):
        if person.sum() == 0:
            continue
        mask = (person.sum(-1) != 0).reshape(T, V, 1)  # only for none zero frames
        skeleton[i_p] = (skeleton[i_p] - main_body_center) * mask
    return skeleton.transpose((3, 1, 2, 0))


def sub_center_jpt_perframe(skeleton, jpt=0, person=0):
    C, T, V, M = skeleton.shape
    skeleton = np.transpose(skeleton, [3, 1, 2, 0])  # M, T, V, C
    main_body_center = skeleton[person, :, jpt].copy().reshape((T, 1, C))  # tc
    for i_p, person in enumerate(skeleton):
        if person.sum() == 0:
            continue
        skeleton[i_p] = (skeleton[i_p] - main_body_center)  # TVC-T1C
    return skeleton.transpose((3, 1, 2, 0))


def obtain_angle(skeleton, edges=()):
    tmp = skeleton.copy()
    for v1, v2 in edges:
        v1 -= 1
        v2 -= 1
        x = skeleton[0, :, v1, :] - skeleton[0, :, v2, :]
        y = skeleton[1, :, v1, :] - skeleton[1, :, v2, :]
        z = skeleton[2, :, v1, :] - skeleton[2, :, v2, :]
        atan0 = np.arctan2(y, x) / 3.14
        atan1 = np.arctan2(z, x) / 3.14
        atan2 = np.arctan2(z, y) / 3.14
        t = np.stack([atan0, atan1, atan2], 0)
        tmp[:, :, v1, :] = t
    return tmp


def norm_len_fstframe(skeleton, jpts=[0, 1], frame=0, person=0):
    C, T, V, M = skeleton.shape
    skeleton = np.transpose(skeleton, [3, 1, 2, 0])  # M, T, V, C
    main_body_spine = np.linalg.norm(skeleton[person, frame, jpts[0]] - skeleton[person, frame, jpts[1]])
    if main_body_spine == 0:
        print('zero bone')
    else:
        skeleton /= main_body_spine
    return skeleton.transpose((3, 1, 2, 0))


####################################################    



def random_move_joint(data_numpy, sigma=0.1):  # 只随机扰动坐标点
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape

    rand_joint = np.random.randn(C, T, V, M) * sigma

    return data_numpy + rand_joint


def pad_recurrent(data):
    skeleton = np.transpose(data, [3, 1, 2, 0])  # C, T, V, M  to  M, T, V, C
    for i_p, person in enumerate(skeleton):
        if person.sum() == 0:
            continue
        if person[0].sum() == 0:  # TVC 去掉头空帧，然后对齐到顶端
            index = (person.sum(-1).sum(-1) != 0)
            tmp = person[index].copy()
            person *= 0
            person[:len(tmp)] = tmp
        for i_f, frame in enumerate(person):
            if frame.sum() == 0:
                if person[i_f:].sum() == 0:  # 循环pad之前的帧
                    rest = len(person) - i_f
                    num = int(np.ceil(rest / i_f))
                    pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                    skeleton[i_p, i_f:] = pad
                    break
    return skeleton.transpose((3, 1, 2, 0))  # ctvm


def pad_recurrent_fix(data, length):  # CTVM
    if data.shape[1] < length:
        num = int(np.ceil(length / data.shape[1]))
        data = np.concatenate([data for _ in range(num)], 1)[:, :length]
    return data


def pad_zero(data, length):
    if data.shape[1] < length:
        new = np.zeros([data.shape[0], length - data.shape[1], data.shape[2], data.shape[3]])
        data = np.concatenate([data, new], 1)
    return data


### sampling
def expand_list(l, length):
    if len(l) < length:
        while len(l) < length:
            tmp = []
            [tmp.extend([x, x]) for x in l]
            l = tmp
        return sample_uniform_list(l, length)
    else:
        return l


def sample_uniform_list(l, length):
    if len(l)==length:
        return l
    interval = len(l) / length
    uniform_list = [int(i * interval) for i in range(length)]
    tmp = [l[x] for x in uniform_list]
    return tmp


def uniform_sample_np(data_numpy, size):
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    interval = T / size
    uniform_list = [int(i * interval) for i in range(size)]
    return data_numpy[:, uniform_list]


def random_sample_np(data_numpy, size):
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    interval = int(np.ceil(size / T))
    random_list = sorted(random.sample(list(range(T))*interval, size))
    return data_numpy[:, random_list]


def random_choose_simple(data_numpy, size, center=False):
    C, T, V, M = data_numpy.shape
    if size < 0:
        assert 'resize shape is not right'
    if T == size:
        return data_numpy
    elif T < size:
        return data_numpy
    else:
        if center:
            begin = (T - size) // 2
        else:
            begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]

def interval_sampling(data_numpy, size, test=False):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if size < 0:
        assert 'resize shape is not right'
    if T == size:
        if test:
            return data_numpy.repeat(repeats=5,axis=3)
        return data_numpy
    elif T < size:
        pad = np.zeros((C, size - T, V, M)).astype(np.float32)
        data_numpy = np.concatenate([data_numpy, pad], axis=1)
        if test:
            return data_numpy.repeat(repeats=5,axis=3)
        return data_numpy
    else:
        ave_duration = T // size
        if test:
            begin = np.random.randint(0, max(1, T - size * ave_duration))
            offsets = np.multiply(list(range(size)), ave_duration) + np.random.randint(ave_duration, size=size) + [begin]*size
            begin = np.random.randint(0, max(1, T - size * ave_duration))
            offsets2 = np.multiply(list(range(size)), ave_duration) + np.random.randint(ave_duration, size=size) + [begin]*size
            begin = np.random.randint(0, max(1, T - size * ave_duration))
            offsets3 = np.multiply(list(range(size)), ave_duration) + np.random.randint(ave_duration, size=size) + [begin]*size
            begin = np.random.randint(0, max(1, T - size * ave_duration))
            offsets4 = np.multiply(list(range(size)), ave_duration) + np.random.randint(ave_duration, size=size) + [begin]*size
            begin = np.random.randint(0, max(1, T - size * ave_duration))
            offsets5 = np.multiply(list(range(size)), ave_duration) + np.random.randint(ave_duration, size=size) + [begin]*size
            data_numpy = np.concatenate((data_numpy[:, offsets],data_numpy[:, offsets2],data_numpy[:, offsets3],data_numpy[:, offsets4],data_numpy[:, offsets5]),-1)
        else:
            begin = np.random.randint(0, max(1, T - size * ave_duration))
            offsets = np.multiply(list(range(size)), ave_duration) + np.random.randint(ave_duration, size=size) + [begin]*size
            data_numpy = data_numpy[:, offsets]
        return data_numpy


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[0.0],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)  # 需要变换的帧的段数 0, 16, 32
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):  # 使得每一帧的旋转都不一样
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])  # xuanzhuan juzhen

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]  # pingyi bianhuan
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_move_whole(data_numpy, agx=0, agy=0, s=1):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    data_numpy = data_numpy.transpose((1, 2, 3, 0)).reshape(-1, C)

    agx = math.radians(agx)
    agy = math.radians(agy)
    Rx = np.asarray([[1, 0, 0], [0, math.cos(agx), math.sin(agx)], [0, -math.sin(agx), math.cos(agx)]])
    Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0, 1, 0], [math.sin(agy), 0, math.cos(agy)]])
    Ss = np.asarray([[s, 0, 0], [0, s, 0], [0, 0, s]])

    data_numpy = np.dot(np.reshape(data_numpy, (-1, 3)), np.dot(Ry, np.dot(Rx, Ss)))
    data_numpy = data_numpy.reshape((T, V, M, C)).transpose((3, 0, 1, 2))
    return data_numpy.astype(np.float32)

####### augs: mirror, reverse #########

def swap_left_right(data, right_chain: list, left_chain: list):
    assert len(data.shape) == 3 and data.shape[-1] == 3 # (num_frames, num_joints, xyz)
    assert len(right_chain) == len(left_chain)
    num_joints = data.shape[1]
    data = data.copy()
    data[..., 0] *= -1
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    return data


def reverse(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3 # (num_frames, num_joints, xyz)
    data = data.copy()
    return np.flip(data,0)