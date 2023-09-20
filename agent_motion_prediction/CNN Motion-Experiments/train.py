# Imports
from loss import *
from utils import *
from config import *
from dataset import *

import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader, random_split

from torch.utils.tensorboard import SummaryWriter


from efficientnet_pytorch import EfficientNet

# ============================================================
# Create model
effnet_encoder = EfficientNet.from_pretrained(config.ENCODER_NAME)
class EffNet_model(nn.Module):
    def __init__(
        self, n_traj=config.N_TRAJ, n_ts = 80, in_channels=[25, 224, 224]):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels[0], out_channels=3, kernel_size=1)
        self.n_traj = n_traj
        self.n_ts = n_ts

        self.encoder_name = config.ENCODER_NAME
        self.encoder = effnet_encoder

        self.conv1 = nn.Conv2d(in_channels=1280, out_channels=512, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1)

        self.conv1_bn = nn.BatchNorm2d(512)
        self.conv2_bn = nn.BatchNorm2d(128)

        for param in list(self.encoder.parameters())[:-5]:
          param.requires_grad = False

        self.act = nn.ReLU()
        self.DO = nn.Dropout(p=0.4)
        # self.fc1 = nn.Linear(62720, 32768)
        # self.bn1 = nn.BatchNorm1d(32768)

        self.fc1 = nn.Linear(3200, 2048)
        self.bn1 = nn.BatchNorm1d(2048)

        # self.fc2 = nn.Linear(32768, 4096)
        # self.bn2 = nn.BatchNorm1d(4096)

        # self.fc3 = nn.Linear(4096, 1024)
        # self.bn3 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)

        self.fc4 = nn.Linear(512, self.n_ts * 2)


    def forward(self, x):
        x_ = self.conv(x)

        latent_vector = self.encoder.extract_features(x_)
        x_ = self.conv1(latent_vector)
        x_ = self.conv1_bn(x_)
        x_ = self.act(x_)

        x_ = self.conv2(x_)
        x_ = self.conv2_bn(x_)
        x_ = self.act(x_)

        x_ = torch.flatten(x_, start_dim=1)

        x_ = self.fc1(x_)
        x_ = self.bn1(x_)
        x_ = self.act(x_)


        x_ = self.fc2(x_)
        x_ = self.bn2(x_)
        x_ = self.act(x_)
        x_ = self.DO(x_)


        x_ = self.fc3(x_)
        x_ = self.bn3(x_)
        x_ = self.act(x_)
        x_ = nn.Dropout(0.1)(x_)


        x_ = self.fc4(x_)

        return x_

def train(batch): 
  x = batch[0].to(config.DEVICE)
  y = batch[1].to(config.DEVICE)
  is_available = batch[2].to(config.DEVICE)

  x = x.reshape(-1, 25, 224, 224)

  optimizer.zero_grad()
  output = eff_model(x)
  loss = loss_func.forward(
          y.view(-1, 80, 2), output.view(-1, 80, 2), is_available
  ).to(config.DEVICE)

  loss.backward()
  optimizer.step()
  scheduler.step()

  # mean square error
  mse = nn.MSELoss()(y.view(-1, 80, 2), output.view(-1, 80, 2))

  return loss, mse

if __name__=="__main__": 
   
  # Create Dataset
  train_set = WaymoDataset(data_path=DIR.RENDER_DIR, type='train')
  val_set = WaymoDataset(data_path=DIR.RENDER_DIR, type='val')
  test_set = WaymoDataset(data_path=DIR.RENDER_DIR, type='test')

  # Create DataLoader
  train_loader = DataLoader(train_set, batch_size=config.TRAIN_BS, shuffle=True)
  val_loader = DataLoader(val_set, batch_size=config.VAL_BS, shuffle=True)
  test_loader = DataLoader(test_set, batch_size=config.TEST_BS, shuffle=True)

  # Create model
  eff_model = EffNet_model()
  eff_model.to(config.DEVICE)

  # Training loop
  loss_func = pytorch_neg_multi_log_likelihood_batch()
  optimizer = torch.optim.AdamW(eff_model.parameters(), lr = config.LR)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
          optimizer,
          T_0=2 * 32,
          T_mult=1,
          eta_min=max(1e-2 * config.LR, 1e-6),
          last_epoch=-1,
  )

  prof = torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(DIR.TB_DIR),
    record_shapes=True,
    with_stack=True)
  
  writer = SummaryWriter(DIR.TB_DIR)
  
  prof.start()

  for epoch in range(config.EPOCHS):
    print(f'{red}{"[INFO]:  "}{res}Epoch {blk}{f"#{epoch+1}/{config.EPOCHS}"}{res} started')

    # Training loop
    for i, data in enumerate(train_loader):
      print("\r", end=f'{progress_bar(i, length=75, train_set_len=len(train_set), train_bs=config.TRAIN_BS)}')

      loss, mse = train(data)
      writer.add_scalar('Loss/train', loss, epoch)
      writer.add_scalar('MSE/train', mse, epoch)

      prof.step()
      # break
    print("Train Loss: {}".format(loss.item()))

    # Validation loop
    with torch.no_grad():
      for i, data in enumerate(val_loader):
        print("\r", end=f'{progress_bar(i, length=75, train_set_len=len(val_set), train_bs=config.VAL_BS)}')

        val_loss, val_mse = train(data)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MSE/val', val_mse, epoch)
        
        # break
      print("Val Loss: {}".format(val_loss.item()))



  prof.stop()