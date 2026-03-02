import argparse
import os

import numpy as np
import torch
import torch.nn as nn

class iprocess(nn.Module):
    def __init__(self):
        super(iprocess, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.PReLU(),
        )
        self.linear_i1 = torch.nn.Linear(in_features=32, out_features=8, bias=True)

    def forward(self, x):
        batch_size = x.shape[0]
        return self.linear_i1(self.cnn1(x).view(batch_size,-1))

class EstimateNow(torch.nn.Module):
    def __init__(self, device):
        super(EstimateNow, self).__init__()
        self.device = device
        self.linear1 = torch.nn.Linear(in_features=2, out_features=8, bias=True)
        self.linear_i1 = torch.nn.Linear(in_features=32, out_features=8, bias=True)
        self.linear_out1 = torch.nn.Linear(in_features=16, out_features=2, bias=True)
        #self.linear3 = torch.nn.Linear(in_features=8, out_features=8, bias=True)
        #self.linear4 = torch.nn.Linear(in_features=8, out_features=8, bias=True)

        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.PReLU(),
        )
        self.ac = nn.Sigmoid()

    def set_input(self, batch):
        cam_1_i = batch['cam1']['img']
        cam_2_i = batch['cam2']['img']
        cam_3_i = batch['cam3']['img']
        cam_4_i = batch['cam4']['img']
        cam_1_xy = torch.stack([batch['cam1']['x'], batch['cam1']['y']], dim=1)
        cam_2_xy = torch.stack([batch['cam2']['x'], batch['cam2']['y']], dim=1)
        cam_3_xy = torch.stack([batch['cam3']['x'], batch['cam3']['y']], dim=1)
        cam_4_xy = torch.stack([batch['cam4']['x'], batch['cam4']['y']], dim=1)
        label_a = batch['cam1']['azimuth'][0].unsqueeze(-1)
        label_p = batch['cam1']['pitch'][0].unsqueeze(-1)
        label_d = batch['cam1']['distance'][0].unsqueeze(-1)
        return [cam_1_i, cam_2_i, cam_3_i, cam_4_i, cam_1_xy, cam_2_xy, cam_3_xy, cam_4_xy], [label_a, label_p, label_d]


    def forward(self, xs):

        cam_1_xy = xs[4].to(self.device).float()
        cam_1_i = xs[0].to(self.device)
        out11 = self.linear_i1(self.cnn1(cam_1_i).view(32, -1))
        out12 = self.linear1(cam_1_xy)
        out13 = torch.cat((out11, out12), dim=1)
        output = self.linear_out1(out13)
        return output

    def loss(self, output, labels):
        label_a, label_p, label_d = labels[0].to(device).float(), labels[1].to(device).float(), labels[2].to(device).float()
        #label = torch.cat((label_a, label_p, label_d), dim=1)
        label = torch.cat((label_a, label_p), dim=1)
        loss = nn.MSELoss()(output, label)
        return loss

    def rmse_db(self, output, labels):
        """
            y_pred, y_true: [b,3]
            返回 [3] 每个维度的 RMSE
            """
        #labels = torch.cat(labels, dim=1).to(self.device)
        label_a, label_p, label_d = labels[0].to(device).float(), labels[1].to(device).float(), labels[2].to(
            device).float()
        # label = torch.cat((label_a, label_p, label_d), dim=1)
        labels = torch.cat((label_a, label_p), dim=1)

        mse = (output - labels) ** 2  # [b,3]
        mse_mean = mse.mean(dim=0)  # 对 batch 取平均 -> [3]
        rmse = torch.sqrt(mse_mean)  # [3]
        rmse_db = 20 * torch.log10(rmse + 1e-8)

        return rmse_db

    def mse(self, output, labels):
        """
            y_pred, y_true: [b,3]
            返回 [3] 每个维度的 RMSE
            """
        label_a, label_p, label_d = labels[0].to(device).float(), labels[1].to(device).float(), labels[2].to(
            device).float()
        # label = torch.cat((label_a, label_p, label_d), dim=1)
        labels = torch.cat((label_a, label_p), dim=1)

        #labels = torch.cat(labels, dim=1).to(self.device)
        mse = abs(output - labels)  # [b,3]
        mse_mean = mse.mean(dim=0)  # 对 batch 取平均 -> [3]
        mse = torch.sqrt(mse_mean)  # [3]
        return mse


def train(args, model, train_loader, test_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input, label = model.set_input(batch)
            optimizer.zero_grad()
            output = model(input)
            loss = model.loss(output, label)
            loss.backward()
            rmse = model.rmse_db(output, label)
            mse = model.mse(output, label)
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  [Train]{epoch}/{args.epochs}  Batch {batch_idx}, Loss = {loss.item():.4f}, RMSE = {rmse.tolist()}, MSE = {mse.tolist()}")
        model.eval()
        total_rmse = torch.zeros(2)
        total_mse = torch.zeros(2)
        n_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                input, label = model.set_input(batch)
                output = model(input)
                loss = model.loss(output, label)
                rmse = model.rmse_db(output, label)
                mse = model.mse(output, label)
                # 变成标量
                # 累加向量
                total_rmse += rmse.cpu()
                total_mse += mse.cpu()
                n_batches += 1
                if batch_idx % 50 == 0:
                    print(f"[Test] Avg Loss = {loss.item():.4f}, RMSE = {rmse.tolist()}, MSE = {mse.tolist()}")

        avg_rmse = (total_rmse / n_batches).tolist()
        avg_mse = (total_mse / n_batches).tolist()

        print("=====================================")
        print(f"Test Avg RMSE (theta, phi) = {avg_rmse}")
        print(f"Test Avg MSE  (theta, phi) = {avg_mse}")

        save_dir = os.path.join(os.getcwd(), f"weights_cam2/e{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")


def test(args, model, test_loader, device):
    model.eval()
    total_rmse = torch.zeros(2)
    total_mse = torch.zeros(2)
    n_batches = 0
    az_diffs = []
    el_diffs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input, label = model.set_input(batch)
            output = model(input)
            loss = model.loss(output, label)
            rmse = model.rmse_db(output, label)
            mse = model.mse(output, label)
            # 变成标量
            az_diff = output[:,0].to('cpu')-label[0].squeeze(-1)
            el_diff = output[:,1].to('cpu')-label[1].squeeze(-1)
            az_diffs.append(az_diff)
            el_diffs.append(el_diff)
            # 累加向量
            total_rmse += rmse.cpu()
            total_mse += mse.cpu()
            n_batches += 1
            if batch_idx % 50 == 0:
                print(f"[Test] Avg Loss = {loss.item():.4f}, RMSE = {rmse.tolist()}, MSE = {mse.tolist()}")

    avg_rmse = (total_rmse / n_batches).tolist()
    avg_mse = (total_mse / n_batches).tolist()

    az_diff_list = torch.cat(az_diffs, dim=-1).tolist()
    el_diff_list = torch.cat(el_diffs, dim=-1).tolist()

    np.save("Cam1_az_diff_list.npy", np.array(az_diff_list))
    np.save("Cam1_el_diff_list.npy", np.array(el_diff_list))


    print("=====================================")
    print(f"Test Avg RMSE (theta, phi) = {avg_rmse}")
    print(f"Test Avg MSE  (theta, phi) = {avg_mse}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)

    params = parser.parse_args()

    #================================ dataset
    from dataset import MultiCamUAVDataset
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
    ])

    dataset = MultiCamUAVDataset(data_root="Data", label_root="label", transform=transform)

    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    test_len = total_len - train_len

    train_set = torch.utils.data.Subset(dataset, list(range(0, train_len)))
    test_set = torch.utils.data.Subset(dataset, list(range(train_len, total_len)))

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = EstimateNow(device).to(device)

    train(params, model, train_loader, test_loader, device)
    #model.load_state_dict(torch.load('/home/wfwu/pythonProj/Genesis_uav/mycode/weights_cam2/e100/model.pth'))
    #model.load_state_dict(torch.load('C:/FengFeng/wfwuCode/Genesis_uav/mycode/weights_cam1/e100/model.pth'))
    test(params, model, test_loader, device)
