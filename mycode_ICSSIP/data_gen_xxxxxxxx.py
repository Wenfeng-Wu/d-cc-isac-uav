import json
import genesis as gs
import numpy as np
import os
import random
import math

import torch
import moviepy
print(moviepy.__version__)

def random_coordinate(range):
    """
    在指定范围内随机生成一个三维坐标
    :param x_range: (xmin, xmax) x 轴范围
    :param y_range: (ymin, ymax) y 轴范围
    :param z_range: (zmin, zmax) z 轴范围
    :return: 生成的 (x, y, z) 坐标
    """
    x = np.random.uniform(*range)
    y = np.random.uniform(*range)
    z = np.random.uniform(*range)
    return (x, y, z)

def random_move(initial_position, step_size, x_range, y_range):
    """
    随机移动物体在 XY 平面上的位置

    :param initial_position: (x, y, z) 初始坐标
    :param step_size: 最大移动步长
    :param x_range: (xmin, xmax) 限制 x 轴范围
    :param y_range: (ymin, ymax) 限制 y 轴范围
    :return: 新的 (x, y, z) 坐标
    """
    x, y, z = initial_position

    # 生成随机移动步长
    dx = random.uniform(-step_size, step_size)
    dy = random.uniform(-step_size, step_size)

    # 计算新坐标，并确保在限定范围内
    new_x = min(max(x + dx, x_range[0]), x_range[1])
    new_y = min(max(y + dy, y_range[0]), y_range[1])

    return new_x, new_y, z

def liner_move2(initial_position, theta, step_size):
    """
    物体沿着指定方向匀速直线运动一步
    :param x: 当前 x 坐标
    :param y: 当前 y 坐标
    :param z: 当前 z 坐标（保持不变）
    :param theta: 运动方向角（相对于 x 轴，单位：度）
    :param step_size: 每次移动的步长
    :return: 更新后的 (x, y, z) 坐标
    """
    x, y, z = initial_position
    theta = np.radians(theta)  # 角度转换为弧度
    x += step_size * np.cos(theta)
    y += step_size * np.sin(theta)
    return (x, y, z)

def liner_move(initial_position, theta, step_size):
    """
    UAV 运动：
    - 水平直线 + 小抖动
    - 高度平滑上下漂移
    - 直接返回 torch.Tensor（GPU）避免后续 stack / numpy 报错
    """

    # ----------- 将输入转 float -----------
    x = float(initial_position[0])
    y = float(initial_position[1])
    z = float(initial_position[2])

    # ----------- 水平方向运动 -----------
    theta_rad = np.radians(float(theta))
    jitter = np.radians(np.random.uniform(-1, 1))  # ±3°抖动
    theta_rad += jitter
    x += step_size * np.cos(theta_rad)
    y += step_size * np.sin(theta_rad)

    # ----------- 垂直方向上下漂移 -----------
    if not hasattr(liner_move, "vz"):
        liner_move.vz = 0.0
    vertical_speed = np.random.uniform(-0.01, 0.01)
    liner_move.vz = 0.9 * liner_move.vz + 0.1 * vertical_speed
    z += liner_move.vz

    # 限制高度
    z = max(1.0, min(z, 2.0))

    # ----------- 返回 GPU Tensor -----------
    return (torch.tensor(x, dtype=torch.float32, device="cuda"),
            torch.tensor(y, dtype=torch.float32, device="cuda"),
            torch.tensor(z, dtype=torch.float32, device="cuda"),)

def calculate_distance(point1, point2):
    """
    计算任意两点之间的欧几里得距离。

    :param point1: 第一个点的坐标 (x1, y1)
    :param point2: 第二个点的坐标 (x2, y2)
    :return: 两点之间的距离
    """
    x1, y1 = point1
    x2, y2 = point2

    # 欧几里得距离公式
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return distance

def euclidean_distance_3d(point1, point2):
    """计算两个三维坐标之间的欧式距离"""
    return math.sqrt((point2[0] - point1[0])**2 +
                     (point2[1] - point1[1])**2 +
                     (point2[2] - point1[2])**2)

def collision(position):
    for i in range(2):
        min_bound, max_bound = range_limitation[i]
        if position[i] <= min_bound or position[i] >= max_bound:
            return True
    return False

def calculate_elevation_angle(reference_point, target_point):
    """
    计算两个三维坐标之间的俯仰角（仰角）。

    参数:
    reference_point (tuple of Tensor): 参考点的三维坐标 (x, y, z)。
    target_point (tuple of Tensor): 目标点的三维坐标 (x, y, z)。

    返回:
    Tensor: 俯仰角（弧度）。
    """
    # 将输入的 tuple 转换为 Tensor
    # reference_point = torch.stack(reference_point)
    target_point = torch.stack(target_point)

    # 计算目标点相对于参考点的向量
    vector = target_point - reference_point

    # 计算水平距离（在 xy 平面上的投影长度）
    horizontal_distance = torch.norm(vector[:2])  # 取 x 和 y 分量

    # 计算垂直距离（z 分量）
    vertical_distance = vector[2]

    # 计算俯仰角（arctan2 可以处理所有象限的情况）
    elevation_angle = torch.atan2(vertical_distance, horizontal_distance)

    return elevation_angle

def calculate_azimuth_angle(reference_point, target_point):
    """
    计算两个三维坐标之间的方位角。

    参数:
    reference_point (tuple of Tensor): 参考点的三维坐标 (x, y, z)。
    target_point (tuple of Tensor): 目标点的三维坐标 (x, y, z)。

    返回:
    Tensor: 方位角（弧度），范围是 [-π, π]。
    """
    # 将输入的 tuple 转换为 Tensor
    # reference_point = torch.stack(reference_point)
    target_point = torch.stack(target_point)

    # 计算目标点相对于参考点的向量
    vector = target_point - reference_point

    # 提取向量的 x 和 y 分量
    dx = vector[0]  # x 分量
    dy = vector[1]  # y 分量

    # 计算方位角（使用 arctan2 计算角度，范围是 [-π, π]）
    azimuth_angle = torch.atan2(dy, dx)

    return azimuth_angle

def calculate_velocity_component(
        stationary_pos,  # 静止目标的坐标 (tuple of Tensor) [x, y, z]
        moving_pos,      # 运动目标的初始坐标 (tuple of Tensor) [x, y, z]
        moving_vel       # 运动目标的速度 (tuple of Tensor) [vx, vy, vz]
):
    """
    计算运动目标在静止目标方向上的速度分量

    参数:
        stationary_pos: 静止目标的坐标 (tuple of Tensor)
        moving_pos: 运动目标的初始坐标 (tuple of Tensor)
        moving_vel: 运动目标的速度 (tuple of Tensor)

    返回:
        速度分量随时间变化的函数（可调用函数，返回 Tensor）
    """
    # 将输入的 tuple 转换为 Tensor
    # stationary_pos = torch.stack(stationary_pos)
    moving_pos = torch.stack(moving_pos)
    # moving_vel = torch.stack(moving_vel)


    # # 确保 t 是 Tensor
    # if not isinstance(t, torch.Tensor):
    #     t = torch.tensor(t, dtype=stationary_pos.dtype, device=stationary_pos.device)
    #
    # # 计算运动目标在时间 t 时的位置
    # current_pos = moving_pos + moving_vel * t

    # 计算从运动目标指向静止目标的向量
    direction_vector = stationary_pos - moving_pos

    # 计算单位向量
    distance = torch.norm(direction_vector)
    if distance == 0:  # 如果两目标重合
        return torch.tensor(0, dtype=stationary_pos.dtype, device=stationary_pos.device)
    unit_vector = direction_vector / distance

    # 计算速度分量（点积）
    #print(moving_vel,unit_vector)
    return torch.dot(moving_vel, unit_vector)


def angle_to_velocity(speed, angle_deg=None, angle_rad=None):
    """
    将方向角和速度值转换为速度向量 [vx, vy, 0]

    参数:
        speed: 速度大小（标量）
        angle_deg: 方向角（角度制，可选）
        angle_rad: 方向角（弧度制，可选）

    返回:
        速度向量 np.array([vx, vy, 0])

    注意:
        必须提供 angle_deg 或 angle_rad 中的一个。
    """
    if angle_deg is not None:
        theta = np.deg2rad(angle_deg)  # 角度转弧度
    elif angle_rad is not None:
        theta = angle_rad
    else:
        raise ValueError("必须提供 angle_deg 或 angle_rad")

    vx = speed * np.cos(theta)
    vy = speed * np.sin(theta)
    return torch.tensor([vx, vy, 0],dtype=torch.float32)

def simulate(iteration):
    def calulate_motion_param(cam_id=0):

        cam_to_drone = euclidean_distance_3d(camera_positions[cam_id], cur_pos_drone)
        record[f"cam_{cam_id}"]["distance"] = [cam_to_drone]

        # print(camera_positions[cam_id], cur_pos_drone)
        cam_to_drone = calculate_elevation_angle(camera_positions[cam_id], cur_pos_drone).cpu().numpy()
        record[f"cam_{cam_id}"]["pitch"] = [cam_to_drone.item()]

        cam_to_drone = calculate_azimuth_angle(camera_positions[cam_id], cur_pos_drone).cpu().numpy()
        record[f"cam_{cam_id}"]["azimuth"] = [cam_to_drone.item()]

        cam_to_drone = calculate_velocity_component(camera_positions[cam_id], cur_pos_drone, drone_moving_velocity).cpu().numpy()
        record[f"cam_{cam_id}"]["radial velocity"] = [cam_to_drone.item()]

    cam_1.start_recording()
    cam_2.start_recording()
    cam_3.start_recording()
    cam_4.start_recording()

    drone_theta = random.choice(thetas)

    records = []

    height = random.choice([i for i in range(1, 6)])

    drone.set_pos((0.0, 0.0, height))

    drone_rate = random.choice(range(20, 60)) / 30
    for i in range(iteration):


        last_pos_drone = drone.get_pos()
        cur_pos_drone = liner_move(last_pos_drone, drone_theta, drone_rate / 100)
        if collision(cur_pos_drone):
            drone_rate*=-1
        drone_moving_velocity = angle_to_velocity(drone_rate, drone_theta).cuda()
        drone.set_pos(cur_pos_drone)


        scene.step()
        cam_1.render()
        cam_2.render()
        cam_3.render()
        cam_4.render()
        ## record
        record = {
         "cam_0":{"distance":[],"azimuth":[],"pitch":[],"radial velocity":[],"rate":[drone_rate]},#[car_rate,dog_rate,drone_rate]},
         "cam_1":{"distance":[],"azimuth":[],"pitch":[],"radial velocity":[],"rate":[drone_rate]}, #[car_rate,dog_rate,drone_rate]},
         "cam_2":{"distance":[],"azimuth":[],"pitch":[],"radial velocity":[],"rate":[drone_rate]}, #[car_rate,dog_rate,drone_rate]},
         "cam_3":{"distance":[],"azimuth":[],"pitch":[],"radial velocity":[],"rate":[drone_rate]}, #[car_rate,dog_rate,drone_rate]}
         }
        for i in range(4):
            calulate_motion_param(i)


        # print(record)
        records.append(record)
    # 保存视频
    os.makedirs("mycode/Data2/cam1",exist_ok=True)
    os.makedirs("mycode/Data2/cam2", exist_ok=True)
    os.makedirs("mycode/Data2/cam3", exist_ok=True)
    os.makedirs("mycode/Data2/cam4", exist_ok=True)
    os.makedirs("mycode/Data2/label", exist_ok=True)
    cam_1.stop_recording(save_to_filename=f'mycode/Data2/cam1/{drone_theta}_{height}.mp4', fps=60)
    cam_2.stop_recording(save_to_filename=f'mycode/Data2/cam2/{drone_theta}_{height}.mp4', fps=60)
    cam_3.stop_recording(save_to_filename=f'mycode/Data2/cam3/{drone_theta}_{height}.mp4', fps=60)
    cam_4.stop_recording(save_to_filename=f'mycode/Data2/cam4/{drone_theta}_{height}.mp4', fps=60)

    # 保存运动参数
    with open(f'mycode/Data/label/{drone_theta}_{height}.json',"w",encoding="utf-8") as f:
        f.write(json.dumps(records,indent=4,ensure_ascii=False))


if __name__ == '__main__':
    plane_range = [()]
    thetas = range(0,360,20)
    camera_positions = [(-3.5,-3.0,0.25), (3.0, -3.5, 0.25), (3.5, 3.0, 0.25), (-3.0, 3.5, 0.25)]
    range_limitation = bounds = [(-1.8, 1.8), (-1.8, 1.8), (5, 15)]  # 物体移动范围
    camera_pos = camera_positions[3]

    ##=============================================  init
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 960),
            camera_pos=camera_pos,
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,  # 是否显示世界坐标系（XYZ 三轴）
            world_frame_size=1.0,   # 世界坐标系的轴长度（单位：米）
            show_link_frame=False,  # 是否显示每个 刚体链接（RigidLink） 的局部坐标系
            link_frame_size=1.0,    # 每个 link 的坐标轴长度
            show_cameras=True,      # 是否渲染场景中的相机及其 视锥体（frustum）
            plane_reflection=False,
            background_color=(0.75, 0.85, 0.95),  # 天空蓝
            ambient_light=(0.1, 0.1, 0.1),
        ),
        renderer=gs.renderers.Rasterizer(),
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    drone = scene.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0.0, 0, 0.0),
            scale=2
        ),
    )

    Obstacle_1 = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 1),
            pos=(-0.6, 0.5, 1),
        )
    )

    Obstacle_3 = scene.add_entity(
        gs.morphs.Box(
            size=(0.6, 0.3, 2.5),
            pos=(0.5, 1.5, 2.5),
        )
    )

    cam_1 = scene.add_camera(
        res=(1280, 960),
        pos=camera_positions[0],
        lookat=(0, 0, 1.5),
        fov=40,
        GUI=False,
    )

    cam_2 = scene.add_camera(
        res=(1280, 960),
        pos=camera_positions[1],
        lookat=(0, 0, 1.5),
        fov=40,
        GUI=False,
    )

    cam_3 = scene.add_camera(
        res=(1280, 960),
        pos=camera_positions[2],
        lookat=(0, 0, 1.5),
        fov=40,
        GUI=False,
    )

    cam_4 = scene.add_camera(
        res=(1280, 960),
        pos=camera_positions[3],
        lookat=(0, 0, 1.5),
        fov=40,
        GUI=False,
    )
    scene.build()
    camera_positions = [torch.tensor(coord, dtype=torch.float32).cuda() for coord in camera_positions]
    for i in range(50):
        simulate(1000)

