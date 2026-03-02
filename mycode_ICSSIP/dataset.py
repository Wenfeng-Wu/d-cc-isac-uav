import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MultiCamUAVDataset_video(Dataset):
    def __init__(self, data_root="Data", label_root="label", transform=None):
        super().__init__()

        self.data_root = data_root
        self.label_root = label_root
        self.transform = transform

        # 文件名，所有 cam1 目录中得到
        self.file_names = [f.replace(".mp4", "") for f in os.listdir(os.path.join(data_root, "cam1")) if
                           f.endswith(".mp4")]
        self.file_names.sort()

        self.num_frames = 500  # 已知每段 500 帧
        self.num_frames_to_use = 300

        # 总长度 = 文件数 × 500
        self.length = len(self.file_names) * self.num_frames_to_use

        self.idx_map = []
        for name in self.file_names:
            for i in range(self.num_frames - self.num_frames_to_use, self.num_frames):
                self.idx_map.append((name, i))  # (视频名, 帧索引)

    def __len__(self):
        return self.length

    def load_frame(self, video_path, frame_idx):
        """读取视频第 frame_idx 帧"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"读视频失败: {video_path}, frame={frame_idx}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def crop_frame(self, frame, x_norm, y_norm, crop_ratio=0.05):
        """
        根据归一化坐标裁剪图像
        :param frame: Tensor 或 np.array [C,H,W] 或 [H,W,C]
        :param x_norm: x 归一化值 [0,1]
        :param y_norm: y 归一化值 [0,1]
        :param crop_ratio: 裁剪比例，相对于原图宽高的百分比
        :return: 裁剪后的图像
        """
        if isinstance(frame, torch.Tensor):
            # Tensor [C,H,W] -> [H,W,C]
            frame_np = frame.permute(1, 2, 0).numpy() * 255
            frame_np = frame_np.astype(np.uint8)
        else:
            frame_np = frame

        H, W = frame_np.shape[:2]

        # (x, y) 归一化坐标转像素坐标
        cx = int(x_norm * W)
        cy = int(y_norm * H)

        # 如果 x,y 都是 0，使用中心点
        if cx == 0 and cy == 0:
            cx, cy = W // 2, H // 2

        # 计算裁剪边界
        w_half = int(W * crop_ratio)
        h_half = int(H * crop_ratio)

        x1 = max(cx - w_half, 0)
        x2 = min(cx + w_half, W)
        y1 = max(cy - h_half, 0)
        y2 = min(cy + h_half, H)

        cropped = frame_np[y1:y2, x1:x2]


        if self.transform:
            frame = self.transform(cropped)
        else:
            frame = torch.tensor(cropped).permute(2, 0, 1) / 255.

        return frame

    def __getitem__(self, idx):
        # 映射到文件名 + 帧号
        #file_idx = idx // self.num_frames
        #frame_idx = self.num_frames - self.num_frames_to_use + idx % self.num_frames_to_use

        #name = self.file_names[file_idx]
        name, frame_idx = self.idx_map[idx]

        # === 读取 4 个摄像头的 x, y json ===
        cam_json = {}
        for cam_id in ["cam1", "cam2", "cam3", "cam4"]:
            json_path = os.path.join(self.data_root, cam_id, name + ".json")
            with open(json_path, "r") as f:
                data = json.load(f)  # data[frame_idx] = {'x':.., 'y':..}
            cam_json[cam_id] = data[str(frame_idx)]

        # === 读取 label（距离、角度、速度） ===
        self.data_root = os.path.join(os.path.dirname(__file__), "Data")
        self.label_dir = os.path.join(self.data_root, "label")
        label_path = os.path.join(self.label_dir, f"{name}.json")
        with open(label_path, "r") as f:
            label_data = json.load(f)  # cam_0..cam_3

        # === 构造每个摄像头的数据 ===
        cams_output = {}
        for cam_id, cam_idx in zip(["cam1", "cam2", "cam3", "cam4"], [0, 1, 2, 3]):
            video_path = os.path.join(self.data_root, cam_id, name + ".mp4")


            xy = cam_json[cam_id]


            img = self.load_frame(video_path, frame_idx)
            img = self.crop_frame(img, xy["x"], xy["y"], crop_ratio=0.1)

            # label_data["cam_0"]["distance"][frame_idx]
            label_cam = label_data[frame_idx][f"cam_{cam_idx}"]

            cams_output[cam_id] = {
                "img": img,
                "x": xy["x"],
                "y": xy["y"],
                "distance": label_cam["distance"],
                "azimuth": label_cam["azimuth"],
                "pitch": label_cam["pitch"],
                "radial_velocity": label_cam["radial velocity"],
                "rate": label_cam["rate"]  # 这是个单值数组
            }

        return {
            "name": name,
            "frame": frame_idx,
            "cam1": cams_output["cam1"],
            "cam2": cams_output["cam2"],
            "cam3": cams_output["cam3"],
            "cam4": cams_output["cam4"],
        }

class MultiCamUAVDataset(Dataset):
    def __init__(self, data_root="Data", label_root="label", transform=None):
        super().__init__()

        self.data_root = data_root
        self.label_root = label_root
        self.transform = transform

        # 文件名，所有 cam1 目录中得到
        self.file_names = [f.replace(".mp4", "") for f in os.listdir(os.path.join(data_root, "cam1")) if
                           f.endswith(".mp4")]
        self.file_names.sort()

        self.num_frames = 500  # 已知每段 500 帧
        self.num_frames_to_use = 300

        # 总长度 = 文件数 × 500
        self.length = len(self.file_names) * self.num_frames_to_use

        self.idx_map = []
        for name in self.file_names:
            for i in range(self.num_frames - self.num_frames_to_use, self.num_frames):
                self.idx_map.append((name, i))  # (视频名, 帧索引)

    def __len__(self):
        return self.length

    def load_frame(self, video_path, frame_idx):
        """读取视频第 frame_idx 帧"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"读视频失败: {video_path}, frame={frame_idx}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def crop_frame(self, frame, x_norm, y_norm, crop_ratio=0.05):
        """
        根据归一化坐标裁剪图像
        :param frame: Tensor 或 np.array [C,H,W] 或 [H,W,C]
        :param x_norm: x 归一化值 [0,1]
        :param y_norm: y 归一化值 [0,1]
        :param crop_ratio: 裁剪比例，相对于原图宽高的百分比
        :return: 裁剪后的图像
        """
        if isinstance(frame, torch.Tensor):
            # Tensor [C,H,W] -> [H,W,C]
            frame_np = frame.permute(1, 2, 0).numpy() * 255
            frame_np = frame_np.astype(np.uint8)
        else:
            frame_np = frame

        H, W = frame_np.shape[:2]

        # (x, y) 归一化坐标转像素坐标
        cx = int(x_norm * W)
        cy = int(y_norm * H)

        # 如果 x,y 都是 0，使用中心点
        if cx == 0 and cy == 0:
            cx, cy = W // 2, H // 2

        # 计算裁剪边界
        w_half = int(W * crop_ratio)
        h_half = int(H * crop_ratio)

        x1 = max(cx - w_half, 0)
        x2 = min(cx + w_half, W)
        y1 = max(cy - h_half, 0)
        y2 = min(cy + h_half, H)

        cropped = frame_np[y1:y2, x1:x2]


        if self.transform:
            frame = self.transform(cropped)
        else:
            frame = torch.tensor(cropped).permute(2, 0, 1) / 255.

        return frame

    def __getitem__(self, idx):
        # 映射到文件名 + 帧号
        #file_idx = idx // self.num_frames
        #frame_idx = self.num_frames - self.num_frames_to_use + idx % self.num_frames_to_use

        #name = self.file_names[file_idx]
        name, frame_idx = self.idx_map[idx]

        # === 读取 4 个摄像头的 x, y json ===
        cam_json = {}
        for cam_id in ["cam1", "cam2", "cam3", "cam4"]:
            json_path = os.path.join(self.data_root, cam_id, name + ".json")
            with open(json_path, "r") as f:
                data = json.load(f)  # data[frame_idx] = {'x':.., 'y':..}
            cam_json[cam_id] = data[str(frame_idx)]

        # === 读取 label（距离、角度、速度） ===
        self.data_root = os.path.join(os.path.dirname(__file__), "Data")
        self.label_dir = os.path.join(self.data_root, "label")
        label_path = os.path.join(self.label_dir, f"{name}.json")
        with open(label_path, "r") as f:
            label_data = json.load(f)  # cam_0..cam_3

        # === 构造每个摄像头的数据 ===
        cams_output = {}
        for cam_id, cam_idx in zip(["cam1", "cam2", "cam3", "cam4"], [0, 1, 2, 3]):
            frames_cache_path = os.path.join("cache", cam_id, f"{name}.npy")
            frames = np.load(frames_cache_path)
            # 这里 idx_map 已经映射到第几帧了
            frame_idx_local = frame_idx - (self.num_frames - self.num_frames_to_use)
            img = frames[frame_idx_local]

            if self.transform:
                img = self.transform(img)
            else:
                img = torch.tensor(img).permute(2, 0, 1) / 255.

            # label_data["cam_0"]["distance"][frame_idx]
            label_cam = label_data[frame_idx][f"cam_{cam_idx}"]
            xy = cam_json[cam_id]
            cams_output[cam_id] = {
                "img": img,
                "x": xy["x"],
                "y": xy["y"],
                "distance": label_cam["distance"],
                "azimuth": label_cam["azimuth"],
                "pitch": label_cam["pitch"],
                "radial_velocity": label_cam["radial velocity"],
                "rate": label_cam["rate"]  # 这是个单值数组
            }

        return {
            "name": name,
            "frame": frame_idx,
            "cam1": cams_output["cam1"],
            "cam2": cams_output["cam2"],
            "cam3": cams_output["cam3"],
            "cam4": cams_output["cam4"],
        }


class MultiCamUAVDataset_time(Dataset):
    def __init__(self, data_root="Data2", label_root="label", transform=None, seq_len=10, frame_step=5):
        super().__init__()

        self.data_root = data_root
        self.label_root = label_root
        self.transform = transform
        self.seq_len = seq_len
        self.frame_step = frame_step

        # 文件名，所有 cam1 目录中得到
        self.file_names = [f.replace(".mp4", "") for f in os.listdir(os.path.join(data_root, "cam1")) if
                           f.endswith(".mp4")]
        self.file_names.sort()

        self.num_frames = 500  # 已知每段 500 帧
        self.num_frames_to_use = 300

        # 构造 idx_map：每个 sample 为连续 seq_len 帧
        self.idx_map = []
        start0 = self.num_frames - self.num_frames_to_use
        max_start = self.num_frames - 1 - (self.seq_len - 1) * self.frame_step
        for name in self.file_names:
            for s in range(start0, max_start + 1):
                self.idx_map.append((name, s))

        self.length = len(self.idx_map)


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        # 映射到文件名 + 帧号
        name, start_idx = self.idx_map[idx]
        end_idx = start_idx + self.seq_len  # 不包含 end_idx

        # === 读取 4 个摄像头的 x, y json ===
        cam_jsons = {}
        for cam_id in ["cam1", "cam2", "cam3", "cam4"]:
            json_path = os.path.join(self.data_root, cam_id, name + ".json")
            with open(json_path, "r") as f:
                cam_jsons[cam_id] = json.load(f)  # data[frame_idx] = {'x':.., 'y':..}

        # ======= 读取 label =======
        label_path = os.path.join(self.data_root, "label", f"{name}.json")
        with open(label_path, "r") as f:
            label_data = json.load(f)

        # ======= 读取 echo label（新增） =======
        echo_path = os.path.join(self.data_root, "label", f"{name}_music_ab-5_p100.json")
        with open(echo_path, "r") as f:
            echo_data = json.load(f)

        # === 构造每个摄像头的数据 ===
        cams_output = {}
        for cam_id, cam_idx in zip(["cam1", "cam2", "cam3", "cam4"], [0, 1, 2, 3]):
            # 加载缓存后的帧序列
            frames_cache_path = os.path.join("cache2", cam_id, f"{name}.npy")
            frames = np.load(frames_cache_path)  # shape [300, H, W, 3]

            start_local = start_idx - (self.num_frames - self.num_frames_to_use)
            #imgs_np = frames[start_local: start_local + self.seq_len]
            end_local = start_local + self.seq_len * self.frame_step
            indices = np.arange(start_local, end_local, self.frame_step)
            imgs_np = frames[indices]

            # transform
            if self.transform:
                imgs = torch.stack([self.transform(img) for img in imgs_np])
            else:
                imgs = torch.tensor(imgs_np).permute(0, 3, 1, 2) / 255.

            # labels
            label_seq = label_data[start_idx: start_idx + self.seq_len]
            distances = [frame[f"cam_{cam_idx}"]["distance"] for frame in label_seq]
            azimuths = [frame[f"cam_{cam_idx}"]["azimuth"] for frame in label_seq]
            pitchs = [frame[f"cam_{cam_idx}"]["pitch"] for frame in label_seq]
            vels = [frame[f"cam_{cam_idx}"]["radial velocity"] for frame in label_seq]
            rates = [frame[f"cam_{cam_idx}"]["rate"] for frame in label_seq]

            # xy for every frame
            xs = [cam_jsons[cam_id][str(i)]["x"] for i in range(start_idx, end_idx)]
            ys = [cam_jsons[cam_id][str(i)]["y"] for i in range(start_idx, end_idx)]

            cams_output[cam_id] = {
                "img": imgs,  # shape: [seq_len, C, H, W]
                "x": torch.tensor(xs),
                "y": torch.tensor(ys),
                "distance": torch.tensor(distances),
                "azimuth": torch.tensor(azimuths),
                "pitch": torch.tensor(pitchs),
                "radial_velocity": torch.tensor(vels),
                "rate": torch.tensor(rates),
            }

        return {
            "name": name,
            "start_frame": start_idx,
            "seq_len": self.seq_len,
            "cam1": cams_output["cam1"],
            "cam2": cams_output["cam2"],
            "cam3": cams_output["cam3"],
            "cam4": cams_output["cam4"],
            "azimuth_echo": torch.tensor(echo_data["azimuth"][start_idx: end_idx]),
            "pitch_echo": torch.tensor(echo_data["pitch"][start_idx: end_idx]),
        }

'''
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
])

dataset = MultiCamUAVDataset_time(data_root="Data", label_root="label", transform=transform)

total_len = len(dataset)
train_len = int(total_len * 0.8)
test_len = total_len - train_len

train_set = torch.utils.data.Subset(dataset, list(range(0, train_len)))
test_set = torch.utils.data.Subset(dataset, list(range(train_len, total_len)))

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False, num_workers=0)

az_min = float('inf')
az_max = float('-inf')
el_min = float('inf')
el_max = float('-inf')

for batch in train_loader:
    cam_1_i = batch['cam1']['img']
    cam_2_i = batch['cam2']['img']
    cam_3_i = batch['cam3']['img']
    cam_4_i = batch['cam4']['img']
    cam_1_xy = torch.stack([batch['cam1']['x'], batch['cam1']['y']], dim=2)
    cam_2_xy = torch.stack([batch['cam2']['x'], batch['cam2']['y']], dim=2)
    cam_3_xy = torch.stack([batch['cam3']['x'], batch['cam3']['y']], dim=2)
    cam_4_xy = torch.stack([batch['cam4']['x'], batch['cam4']['y']], dim=2)
    azimuth_echo = batch['azimuth_echo']
    elevation_echo = batch['pitch_echo']
    azimuth_true = batch['cam1']['azimuth'].squeeze(-1)
    elevation_true = batch['cam1']['pitch'].squeeze(-1)
    # 更新全局 min/max
    az_min = min(az_min, azimuth_true.min().item())
    az_max = max(az_max, azimuth_true.max().item())
    el_min = min(el_min, elevation_true.min().item())
    el_max = max(el_max, elevation_true.max().item())

print("Azimuth   min/max:", az_min, az_max)
print("Elevation min/max:", el_min, el_max)


az_min = float('inf')
az_max = float('-inf')
el_min = float('inf')
el_max = float('-inf')

for batch in test_loader:
    cam_1_i = batch['cam1']['img']
    cam_2_i = batch['cam2']['img']
    cam_3_i = batch['cam3']['img']
    cam_4_i = batch['cam4']['img']
    cam_1_xy = torch.stack([batch['cam1']['x'], batch['cam1']['y']], dim=2)
    cam_2_xy = torch.stack([batch['cam2']['x'], batch['cam2']['y']], dim=2)
    cam_3_xy = torch.stack([batch['cam3']['x'], batch['cam3']['y']], dim=2)
    cam_4_xy = torch.stack([batch['cam4']['x'], batch['cam4']['y']], dim=2)
    azimuth_echo = batch['azimuth_echo']
    elevation_echo = batch['pitch_echo']
    azimuth_true = batch['cam1']['azimuth'].squeeze(-1)
    elevation_true = batch['cam1']['pitch'].squeeze(-1)
    # 更新全局 min/max
    az_min = min(az_min, azimuth_true.min().item())
    az_max = max(az_max, azimuth_true.max().item())
    el_min = min(el_min, elevation_true.min().item())
    el_max = max(el_max, elevation_true.max().item())

print("Azimuth   min/max:", az_min, az_max)
print("Elevation min/max:", el_min, el_max)

'''