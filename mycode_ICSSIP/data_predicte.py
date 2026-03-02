import argparse
import math
import os

import torch
import torch.nn as nn

from Genesis_uav.mycode_ICSSIP.cul_r_throughout import cul_through_out
from data_allignment import EstimateNow

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        return x + self.pe[:, :T, :D]

class PredictNext(nn.Module):
    def __init__(self, time_slot, device):
        super().__init__()
        self.device = device
        self.time_slot = time_slot

        self.az_max = 1.15
        self.az_min = 0.3
        self.az_range = self.az_max - self.az_min

        self.el_max = 0.6
        self.el_min = 0.1
        self.el_range = self.el_max - self.el_min

        input_dim = 2
        output_dim = 2
        model_dim = 8
        num_heads = 4
        num_layers = 2
        # 1. 输入映射到 model_dim
        self.input_proj = nn.Linear(input_dim, model_dim)

        # 2. 位置编码（可学习）
        self.pos_embed = nn.Parameter(torch.randn(1, 100, model_dim))  # 支持最长500步

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 输出层（预测下一时刻2维）
        self.fc = nn.Sequential(nn.Linear(model_dim, output_dim),
                                nn.Sigmoid())

        self.ac = nn.Sigmoid()
        self.weight = nn.Parameter(torch.rand(self.time_slot))

    def forward(self, cams_xy, cam_est_azel, echo_est_azel):
        # cams_xy shape: list (batch, seq_len, 2) * 4
        # cam_est_azel shape: (batch, seq_len, 2)
        # echo_est_azel shape: (batch, seq_len, 2)
        echo_est_azel = echo_est_azel[:, :-1, :].to(self.device)
        B, T, _ = echo_est_azel.shape
        # 输入映射 + 位置编码
        x_proj = self.input_proj(echo_est_azel) #+ self.pos_embed[:, :T]
        h = self.transformer(x_proj)  # [B, T, model_dim]
        last = h[:, -1]  # [B, model_dim]
        next_state = self.fc(last)  # [B, 2]
        return echo_est_azel[:,-1,:]+(next_state*0.3)-0.15


    def set_input(self, batch):
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
        #azimuth_true = (azimuth_true - torch.min(azimuth_true)) / (torch.max(azimuth_true) - torch.min(azimuth_true))
        #elevation_true = (elevation_true - torch.min(elevation_true)) / (torch.max(elevation_true) - torch.min(elevation_true))

        return ([cam_1_i.flatten(0,1), cam_2_i.flatten(0,1), cam_3_i.flatten(0,1), cam_4_i.flatten(0,1),
                cam_1_xy.flatten(0,1), cam_2_xy.flatten(0,1), cam_3_xy.flatten(0,1), cam_4_xy.flatten(0,1)],
                [cam_1_xy, cam_2_xy, cam_3_xy, cam_4_xy],
                torch.stack([azimuth_echo, elevation_echo], dim=-1),
                torch.stack([azimuth_true, elevation_true], dim=-1))

    def loss(self, x, y):
        #y = y[:,0].unsqueeze(-1)
        loss_fn = nn.MSELoss()
        return loss_fn(x, y.to(self.device))

    def rmse_db(self, x, y):
        #mse = (x - y[:,0].unsqueeze(-1).to(self.device)) ** 2  # [b,3]
        mse = (x - y.to(self.device)) ** 2  # [b,3]
        mse_mean = mse.mean(dim=0)  # 对 batch 取平均 -> [3]
        rmse = torch.sqrt(mse_mean)  # [3]
        rmse_db = 20 * torch.log10(rmse + 1e-8)

        return rmse_db

    def mse(self, x, y):
        #mse = (x - y[:,0].unsqueeze(-1).to(self.device)) ** 2  # [b,3]
        mse = (x - y.to(self.device)) ** 2  # [b,3]
        mse_mean = mse.mean(dim=0)  # 对 batch 取平均 -> [3]
        mse = torch.sqrt(mse_mean)  # [3]
        return mse

def set_model_pre_input(x):
    return


def train(args, model, model_pre, train_loader, test_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model_pre.to(device)
    model_pre.eval()
    for p in model_pre.parameters():
        p.requires_grad = False

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            premodel_input, model_input_cam, model_input_echo, label = model.set_input(batch)

            #cam_est_azel = model_pre(premodel_input)
            b = model_input_cam[0].shape[0]
            t = model_input_cam[0].shape[1]
            #cam_est_azel = cam_est_azel.unflatten(0,(b,t))

            output = model(model_input_cam, label, label)
            loss = model.loss(output, label[:,-1,:])
            loss.backward()
            optimizer.step()

            rmse = model.rmse_db(output, label[:,-1,:])
            mse = model.mse(output, label[:,-1,:])
            total_loss += loss.item()



            if batch_idx % 1 == 0:
                print(f"  [Train]{epoch}/{args.epochs}  Batch {batch_idx}, Loss = {loss.item():.4f}, RMSE = {rmse.tolist()}, MSE = {mse.tolist()}")

        '''
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                premodel_input, model_input_cam, model_input_echo, label = model.set_input(batch)

                cam_est_azel = model_pre(premodel_input)
                b = model_input_cam[0].shape[0]
                t = model_input_cam[0].shape[1]
                cam_est_azel = cam_est_azel.unflatten(0, (b, t))

                output = model(model_input_cam, cam_est_azel, cam_est_azel)
                loss = model.loss(output, label[:,-1,:])
                rmse = model.rmse_db(output, label[:,-1,:])
                mse = model.mse(output, label[:,-1,:])
                print(f"[Test] Avg Loss = {loss.item():.4f}, RMSE = {rmse.tolist()}, MSE = {mse.tolist()}")
                break
                '''
        save_dir = os.path.join(os.getcwd(), f"weights_pre_mm2/e{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")
        if epoch % 5 == 0:
            model.eval()
            total_rmse = torch.zeros(2)
            total_mse = torch.zeros(2)
            n_batches = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    premodel_input, model_input_cam, model_input_echo, label = model.set_input(batch)

                    cam_est_azel = model_pre(premodel_input)
                    b = model_input_cam[0].shape[0]
                    t = model_input_cam[0].shape[1]
                    cam_est_azel = cam_est_azel.unflatten(0, (b, t))

                    output_cam = model(model_input_cam, cam_est_azel, cam_est_azel)
                    rmse = model.rmse_db(output_cam, label[:, -1, :])
                    mse = model.mse(output_cam, label[:, -1, :])

                    total_rmse += rmse.cpu()
                    total_mse += mse.cpu()
                    n_batches += 1

            avg_rmse = (total_rmse / n_batches).tolist()
            avg_mse = (total_mse / n_batches).tolist()

            print("=====================================")
            print(f"Test Avg RMSE (theta, phi) = {avg_rmse}")
            print(f"Test Avg MSE  (theta, phi) = {avg_mse}")


def test(params, model, model_pre, test_loader, device):
    model_pre.eval()
    model_pre.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        total_rmse = torch.zeros(2)
        total_rmse2 = torch.zeros(2)
        total_mse = torch.zeros(2)
        total_mse2 = torch.zeros(2)
        total_R = 0
        total_R2 = 0
        n_batches = 0
        for batch_idx, batch in enumerate(test_loader):
            premodel_input, model_input_cam, model_input_echo, label = model.set_input(batch)

            cam_est_azel = model_pre(premodel_input)
            b = model_input_cam[0].shape[0]
            t = model_input_cam[0].shape[1]
            cam_est_azel = cam_est_azel.unflatten(0, (b, t))

            output_cam = model(model_input_cam, cam_est_azel, model_input_echo)
            output_cam2 = model(model_input_cam, cam_est_azel, cam_est_azel)
            rmse = model.rmse_db(output_cam, label[:, -1, :])
            rmse2 = model.rmse_db(output_cam2, label[:, -1, :])
            mse = model.mse(output_cam, label[:, -1, :])
            mse2 = model.mse(output_cam2, label[:, -1, :])
            rs = 0
            rs2 = 0
            B = output_cam.shape[0]
            for i in range(B):
                r = cul_through_out(torch.cat([output_cam[0,:].to('cpu'), label[0,-1,:]], dim=0).tolist())
                r2 = cul_through_out(torch.cat([output_cam2[0,:].to('cpu'), label[0,-1,:]], dim=0).tolist())
                rs = rs+r
                rs2 = rs2+r2

            total_rmse += rmse.cpu()
            total_rmse2 += rmse2.cpu()
            total_mse += mse.cpu()
            total_mse2 += mse2.cpu()
            total_R = total_R + rs/B
            total_R2 = total_R2 + rs2/B
            n_batches += 1

        avg_rmse = (total_rmse / n_batches).tolist()
        avg_rmse2 = (total_rmse2 / n_batches).tolist()
        avg_mse = (total_mse / n_batches).tolist()
        avg_mse2 = (total_mse2 / n_batches).tolist()
        avg_r = total_R / n_batches
        avg_r2 = total_R2 / n_batches


        print("=====================================")
        print(f"Test Avg RMSE (theta, phi) = {avg_rmse}")
        print(f"Test Avg RMSE2 (theta, phi) = {avg_rmse2}")
        print(f"Test Avg MSE  (theta, phi) = {avg_mse}")
        print(f"Test Avg MSE2  (theta, phi) = {avg_mse2}")
        print(f"Test Avg R  (theta, phi) = {avg_r}")
        print(f"Test Avg R2  (theta, phi) = {avg_r2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--time_slot", type=int, default=12)

    params = parser.parse_args()

    # ================================ dataset
    from dataset import MultiCamUAVDataset_time
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
    ])

    dataset = MultiCamUAVDataset_time(data_root="Data2", label_root="label", transform=transform, seq_len=params.time_slot+1, frame_step=1)

    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    test_len = total_len - train_len

    train_set = torch.utils.data.Subset(dataset, list(range(0, train_len)))
    test_set = torch.utils.data.Subset(dataset, list(range(train_len, total_len)))

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = PredictNext(params.time_slot, device).to(device)
    model_pre = EstimateNow(device)

    #model_pre.load_state_dict(torch.load('/home/wfwu/pythonProj/Genesis_uav/mycode/weights_camAll/e100/model.pth'))
    model_pre.load_state_dict(torch.load('C:/FengFeng/wfwuCode/Genesis_uav/mycode/weights_camAll/e100/model.pth'))

    #train(params, model, model_pre, train_loader, test_loader, device)

    model.load_state_dict(torch.load('C:/FengFeng/wfwuCode/Genesis_uav/mycode/weights_pre_mm2/e0/model.pth'))

    test(params, model, model_pre, test_loader, device)
