import cv2
import numpy as np
import os
import json
import torch

# 背景减除器
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)

def process_video(video_path):
    """处理单个视频，返回 (tensor, json_dict)"""
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    json_dict = {}   # { frame_idx : {'x': x_norm , 'y': y_norm} }
    tensor_list = [] # [[frame_idx, x_norm, y_norm], ...]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cx, cy = 0, 0
        fgmask = fgbg.apply(frame)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

        # 归一化
        x_norm = cx / width
        y_norm = cy / height

        # 保存为 JSON 格式
        json_dict[frame_idx] = {
            'x': float(x_norm),
            'y': float(y_norm)
        }

        # 保存 tensor 格式
        tensor_list.append([frame_idx, x_norm, y_norm])

        frame_idx += 1

    cap.release()

    # 转 tensor
    tensor_out = torch.tensor(tensor_list, dtype=torch.float32)
    return tensor_out, json_dict


# -------------------------------
# 批量处理当前文件夹所有视频
# -------------------------------
video_ext = [".mp4"]
videos = [f for f in os.listdir(".") if os.path.splitext(f)[1].lower() in video_ext]

print("检测到视频文件：", videos)

for vid in videos:
    print(f"\n处理视频：{vid}")

    tensor_out, json_dict = process_video(vid)

    # 保存 JSON（dict 格式）
    json_name = os.path.splitext(vid)[0] + ".json"
    with open(json_name, "w") as f:
        json.dump(json_dict, f, indent=2)

    print(f"JSON 已保存到：{json_name}")
    print(f"Tensor shape: {tensor_out.shape}")
