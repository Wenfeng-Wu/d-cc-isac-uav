import matplotlib.pyplot as plt
import numpy as np
##===================== RMSE 的柱状图对比 =================##
# 数据
groups = ['A', 'B', 'C', 'D']
azimuth = [-32.5916, -17.2426, -36.3659, -29.583]
elevation = [-28.7682, -26.4863, -32.9346, -27.672]


x = np.arange(len(groups))  # 横坐标位置
width = 0.35  # 柱宽

fig, ax = plt.subplots(figsize=(6, 2))

# 绘制柱状图
rects1 = ax.bar(x - width/2, azimuth, width, label='Azimuth', color='#FCD5B5')
rects2 = ax.bar(x + width/2, elevation, width, label='Elevation', color='#DBEEF4')

# 横坐标刻度
ax.set_xticks(x)
ax.set_xticklabels(groups)

# 设置坐标轴标签及字体大小
ax.set_xlabel(' ', fontsize=14)   # 横坐标名称
ax.set_ylabel('RMSE(dB)', fontsize=14)   # 纵坐标名称
ax.set_ylim(-38,-15)
# 设置刻度字体大小
ax.tick_params(axis='x', labelsize=12)  # 横坐标刻度字体大小
ax.tick_params(axis='y', labelsize=12)  # 纵坐标刻度字体大小

# 图例
ax.legend(fontsize=12, loc='upper right')
plt.subplots_adjust(bottom=0.2)
plt.show()

##=====================训练epoch & Loss=======================##

import re
import json

log_file = "C:/FengFeng/wfwuCode/Genesis_uav/mycode_ICSSIP/weights_cam2/cam2_train_record"

# 匹配 [Train] epoch/batch Loss=xx
train_pat = re.compile(r"\[Train\](\d+)/100.*Loss\s*=\s*([0-9.]+)")

# 匹配 [Test] Avg Loss=xx
test_loss_pat = re.compile(r"\[Test\]\s*Avg Loss\s*=\s*([0-9.]+)")

epoch_results = {}

with open(log_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

current_epoch = None
test_loss_buffer = []

for line in lines:

    # -------------------------
    # 解析 Train 行（每个 batch 一条）
    # -------------------------
    m_train = train_pat.search(line)
    if m_train:
        epoch = int(m_train.group(1))
        loss = float(m_train.group(2))

        epoch_results.setdefault(epoch, {})
        epoch_results[epoch].setdefault("train_loss_list", [])
        epoch_results[epoch]["train_loss_list"].append(loss)

        current_epoch = epoch
        continue

    # -------------------------
    # 解析 Test Avg Loss（每个 epoch 2 条）
    # -------------------------
    m_test = test_loss_pat.search(line)
    if m_test:
        test_loss = float(m_test.group(1))
        if current_epoch is not None:
            test_loss_buffer.append(test_loss)

            if len(test_loss_buffer) == 2:
                epoch_results[current_epoch]["test1_loss"] = test_loss_buffer[0]
                epoch_results[current_epoch]["test2_loss"] = test_loss_buffer[1]
                test_loss_buffer = []
        continue

# -------------------------
# 计算 epoch 平均 loss
# -------------------------
for e, info in epoch_results.items():
    if "train_loss_list" in info:
        lst = info["train_loss_list"]
        info["train_loss_mean"] = sum(lst) / len(lst)

# 保存 JSON
with open("epoch_loss_cam2.json", "w") as f:
    json.dump(epoch_results, f, indent=4)

print("已保存 epoch_loss_cam2.json")


import json
import matplotlib.pyplot as plt

# 3 个 JSON 文件
json_files = [
    "epoch_loss_cam1.json",
    "epoch_loss_cam2.json",
    "epoch_loss_camAll.json"
]

plt.figure(figsize=(6, 4))
mak = ['o', 's', '^']
label = ['Cam.1', 'Cam.2', 'Cam.1-Cam.4']
i=0
for jf in json_files:

    with open(jf, "r") as f:
        data = json.load(f)

    # 转换 key 为整数 epoch
    data = {int(k): v for k, v in data.items()}
    epochs = sorted(data.keys())

    # 把所有 epoch 的 batch loss 拼成一条曲线
    batch_losses = []
    for e in epochs:
        batch_losses.extend(data[e]["train_loss_list"])

    label_prefix = jf.replace(".json", "")

    plt.plot(batch_losses, marker=mak[i], label=label[i])
    i = i + 1

plt.yscale("log")
plt.xlabel(r"Batch Index ($\times$ 50)", fontsize=14)
plt.ylabel("Loss", fontsize=14)
#plt.title("Batch-wise Training Loss (Cam1, Cam2, CamAll)", fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.xlim(2, 50)
plt.ylim(0, 0.016)
plt.tight_layout()
plt.show()

