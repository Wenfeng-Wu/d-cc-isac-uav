import os, json, cv2, numpy as np

data_root = "Data2"
cache_root = "cache2"
num_frames_to_use = 900
crop_ratio = 0.05

os.makedirs(cache_root, exist_ok=True)

for cam_id in ["cam1", "cam2", "cam3", "cam4"]:
    cam_path = os.path.join(data_root, cam_id)
    cache_path = os.path.join(cache_root, cam_id)
    os.makedirs(cache_path, exist_ok=True)
    TARGET_SIZE = (96, 96)

    for video_file in os.listdir(cam_path):
        if not video_file.endswith(".mp4"):
            continue
        name = video_file.replace(".mp4", "")
        print(f"Processing {cam_id} - {name}")
        video_path = os.path.join(cam_path, video_file)
        json_path = os.path.join(cam_path, name + ".json")

        # 读取 JSON 坐标
        with open(json_path, "r") as f:
            coords = json.load(f)

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_idx = max(0, total_frames - num_frames_to_use)
        frames_list = []

        for i in range(start_idx, total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"读视频失败: {video_path}, frame={i}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 裁剪
            xy = coords[str(i)]
            H, W = frame.shape[:2]
            if xy["x"]==0:
                print("no uav seen")
            cx = int(xy["x"] * W) or W // 2
            cy = int(xy["y"] * H) or H // 2
            w_half = int(W * crop_ratio * 0.75)
            h_half = int(H * crop_ratio)
            x1 = max(cx - w_half, 0)
            x2 = min(cx + w_half, W)
            y1 = max(cy - h_half, 0)
            y2 = min(cy + h_half, H)
            cropped = frame[y1:y2, x1:x2]
            crop = cv2.resize(cropped, TARGET_SIZE)

            if crop.size == 0 or len(crop.shape) != 3:
                print("Bad crop at frame", i, "shape:", crop.shape)
                continue

            frames_list.append(crop)

        cap.release()
        frames_array = np.array(frames_list)  # shape [num_frames_to_use, H_crop, W_crop, 3]
        np.save(os.path.join(cache_path, f"{name}.npy"), frames_array)
