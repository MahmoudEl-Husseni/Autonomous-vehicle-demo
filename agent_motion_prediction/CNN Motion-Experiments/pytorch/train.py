# Imports
from loss import *
from utils import *
from config import *
from dataset import *
from visualize import *

import time
# time.sleep(30)

import os
import cv2
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader, random_split

from torch.utils.tensorboard import SummaryWriter


from efficientnet_pytorch import EfficientNet

# ============================================================
# Create model
# ============================================================

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

        self.fc1 = nn.Linear(3200, 2048)
        self.bn1 = nn.BatchNorm1d(2048)

        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        #                             6 * 80 * 2 + 6
        self.fc4 = nn.Linear(512, self.n_traj * self.n_ts * 2 + self.n_traj)


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
        
        if x_.shape[0] > 1:
          x_ = self.bn1(x_)

        x_ = self.act(x_)

        x_ = self.fc2(x_)
        
        if x_.shape[0] > 1:
          x_ = self.bn2(x_)

        x_ = self.act(x_)
        x_ = self.DO(x_)


        x_ = self.fc3(x_)
        if x_.shape[0] > 1:
          x_ = self.bn3(x_)
        x_ = self.act(x_)
        x_ = nn.Dropout(0.1)(x_)


        x_ = self.fc4(x_)

        return x_
    
def load_checkpoint(checkpoint, model, optimizer):
  """Loads a checkpoint into a model and optimizer.

  Args:
    checkpoint: The path to the checkpoint file.
    model: The model to load the checkpoint into.
    optimizer: The optimizer to load the checkpoint into.

  Returns:
    None.
  """

  print(f"{red}{'[INFO]:  '}{res}Loading checkpoint...")
  model.load_state_dict(checkpoint["state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer"])
  print(f"{red}{'[INFO]:  '}{res}Checkpoint loaded")

def train(batch, train=True, loss_ciretria='neg_multi_log_likelihood'): 
  x = batch[0].to(config.DEVICE)
  y = batch[1].to(config.DEVICE).view(-1, 2, 80)
  is_available = batch[2].to(config.DEVICE).view(-1, 1, 80)

  x = x.reshape(-1, 25, 224, 224)

  optimizer.zero_grad()
  output = eff_model(x)
  y_pred, confidences = output[:, :-config.N_TRAJ], output[:, -config.N_TRAJ:]
  confidences = confidences.view(-1, 1, config.N_TRAJ)
  if loss_ciretria == 'neg_multi_log_likelihood':
    y_pred = y_pred.view(-1, config.N_TRAJ, 2, config.FUTURE_TS)
    loss = loss_func.forward(
            y.view(-1, 2, 80), y_pred, confidences, is_available
    ).to(config.DEVICE)

  elif loss_ciretria == 'log_mean_displacement_error':
    y_pred = y_pred.view(-1, config.FUTURE_TS, 2)
    loss = loss_func.forward(
            y.view(-1, 80, 2), y_pred, is_available
    ).to(config.DEVICE)

  if train:
    loss.backward()
    optimizer.step()
    scheduler.step()

  best_traj_idx = torch.argmax(confidences, dim=2).view(-1)
  y_pred = y_pred.view(-1, config.N_TRAJ, config.FUTURE_TS, 2)
  y_pred_best = torch.zeros_like(y_pred[:, 0, :, :])
  for i in range(y_pred_best.shape[0]):
    y_pred_best[i] = y_pred[i, best_traj_idx[i], :, :]
  
  # mean square error
  mse = nn.MSELoss()(y.view(-1, config.FUTURE_TS, 2), y_pred_best.view(-1, config.FUTURE_TS, 2))

  # mean displacement error
  mde = mean_displacement_error(y.view(-1, 80, 2), y_pred_best.view(-1, 80, 2), is_available.view(-1, 80, 1))

  # final displacement error
  fde = final_displacement_error(y.view(-1, 80, 2), y_pred_best.view(-1, 80, 2), is_available.view(-1, 80, 1))
  return loss, mse, mde, fde

def vis_model(model, dataset, epoch, t='train', use_top1=True, save=False): 
  with torch.no_grad():
    for i in range(4):
      data = dataset[i][0]
      output = model(torch.Tensor(data['raster']).to(config.DEVICE).reshape(-1, 25, 224, 224))
      logits, confidences = output[:, :-config.N_TRAJ], output[:, -config.N_TRAJ:]

      logits = logits.view(config.N_TRAJ, config.FUTURE_TS, 2)
      confidences = confidences.view(config.N_TRAJ, 1)

      fig_path = os.path.join(DIR.VIS_DIR, f"epoch_{epoch}_{t}_{i}.png")
      plot_pred(data, logits, confidences, use_top1=use_top1, save=save, save_path=fig_path)
      img = cv2.imread(fig_path)
      img = cv2.resize(img, (config.VIS_HEIGHT, config.VIS_WIDTH))
      writer.add_image(f"{t}_{epoch}_{i}", img.transpose(2, 0, 1), 0)
      if i == 4:
        break


if __name__=="__main__": 

  # Create Dataset
  train_set = WaymoDataset(data_path=DIR.RENDER_DIR, type='train')
  val_set = WaymoDataset(data_path=DIR.RENDER_DIR, type='val')
  test_set = WaymoDataset(data_path=DIR.RENDER_DIR, type='test')

  vis_train_set = WaymoDataset(data_path=DIR.RENDER_DIR, type='train', vis=True)
  vis_val_set = WaymoDataset(data_path=DIR.RENDER_DIR, type='val', vis=True)

  # Create DataLoader
  train_loader = DataLoader(train_set, batch_size=config.TRAIN_BS, shuffle=True)
  val_loader = DataLoader(val_set, batch_size=config.VAL_BS, shuffle=True)
  test_loader = DataLoader(test_set, batch_size=config.TEST_BS, shuffle=True)

  # Create model
  eff_model = EffNet_model()
  eff_model.to(config.DEVICE)
  eff_model = nn.DataParallel(eff_model)

  # Create loss function and optimizer
  if config.LOSS == 'neg_multi_log_likelihood':
    loss_func = pytorch_neg_multi_log_likelihood_batch()
  elif config.LOSS == 'log_mean_displacement_error':
    loss_func = pytorch_log_mean_displacement_error()
  
  optimizer = torch.optim.AdamW(eff_model.parameters(), lr = config.LR)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
          optimizer,
          T_0=2 * 32,
          T_mult=1,
          eta_min=max(1e-2 * config.LR, 1e-6),
          last_epoch=-1,
  )

  # Check if model is already trained
  if os.path.exists(os.path.join(DIR.OUT_DIR, "last_model.pth")):
    config.LOAD_MODEL = True
  
  # Continue from last checkpoint
  if config.LOAD_MODEL:
    load_checkpoint(torch.load(os.path.join(DIR.CKPT_DIR, "last_model.pth")), eff_model, optimizer)

  # get number of last epoch and min loss
  if os.path.exists(os.path.join(DIR.OUT_DIR, "logs.csv")):
    with open(os.path.join(DIR.OUT_DIR, "logs.csv"), "r") as f:
      last_epoch=-1
      for line in f.readlines():
        last_epoch += 1
  else:
    last_epoch = 0
  
  if os.path.exists(os.path.join(DIR.OUT_DIR, "best_logs.csv")):
    with open(os.path.join(DIR.OUT_DIR, "best_logs.csv"), "r") as f:
      n = 0
      for line in f.readlines():
        n += 1
      best_loss = float(line.split(",")[1])
  else:
    best_loss = 1e9

  # Create csv files
  if not os.path.exists(DIR.OUT_DIR):
    os.makedirs(DIR.OUT_DIR)
  if not os.path.exists(DIR.CKPT_DIR):
    os.makedirs(DIR.CKPT_DIR)
  if not os.path.exists(DIR.TB_DIR):
    os.makedirs(DIR.TB_DIR)
  if not os.path.exists(DIR.VIS_DIR):
    os.makedirs(DIR.VIS_DIR)
  
  logging.basicConfig(filename=os.path.join(DIR.OUT_DIR, 'train.log'), encoding='utf-8', level=logging.DEBUG, format='%(levelname)s:%(asctime)s %(message)s')
  if not os.path.exists(os.path.join(DIR.OUT_DIR, "logs.csv")):
    with open(os.path.join(DIR.OUT_DIR, "logs.csv"), "w") as f:
      f.write("epoch,loss,mse,mde,fde\n")
  if not os.path.exists(os.path.join(DIR.OUT_DIR, "best_logs.csv")):
    with open(os.path.join(DIR.OUT_DIR, "best_logs.csv"), "w") as f:
      f.write("epoch,loss,mse,mde,fde\n")
  
  # Create tensorboard writer
  prof = torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(DIR.TB_DIR),
    record_shapes=True,
    with_stack=True)
  
  writer = SummaryWriter(DIR.TB_DIR)
  
  prof.start()

  # Training loop
  for epoch in range(last_epoch, config.EPOCHS):
    print(f'{red}{"[INFO]:  "}{res}Epoch {blk}{f"#{epoch+1}/{config.EPOCHS}"}{res} started')

    # Training loop
    try: 
      for i, data in enumerate(train_loader):
        print("\r", end=f'{progress_bar(i, length=75, train_set_len=len(train_set), train_bs=config.TRAIN_BS)}')
        logging.info(f"Training Epoch {epoch+1}, batch {i+1} / {len(train_loader)}")

        try:
          loss, mse, mde, fde = train(data)
        except Exception as e:
          logging.error(e.__str__() + "Error in training, in Line: " + str(e.__traceback__.tb_lineno))
        

        try: 
          writer.add_scalar('Loss/train', loss, epoch)
          writer.add_scalar('MSE/train', mse, epoch)
          writer.add_scalar('MDE/train', mde, epoch)
          writer.add_scalar('FDE/train', fde, epoch)
        except Exception as e:
          logging.error(e.__str__() + "Error in writing to tensorboard, in Line " + str(e.__traceback__.tb_lineno))
        
        prof.step()
        
        # if i >= 1:
        #   break

      # Visualize train
      try : 
        vis_model(eff_model, vis_train_set, epoch, t='train', use_top1=True, save=True)

        # Visualize val
        vis_model(eff_model, vis_val_set, epoch, t='val', use_top1=False, save=True)
      except Exception as e:
        logging.error(e.__str__() + "in Line " + str(e.__traceback__.tb_lineno))

      print("Train Loss: {}".format(loss.item()))


      # write logs to csv file
      with open(os.path.join(DIR.OUT_DIR, "logs.csv"), "a") as f:
        f.write(f"{epoch+1},{loss.item()},{mse.item()},{mde.item()},{fde.item()}\n")

      # Checkpoint loop
      if epoch % config.CKPT_EPOCH == 0:
        torch.save(eff_model.state_dict(), os.path.join(DIR.CKPT_DIR, f"model_{epoch}.pth"))

      # Save checkpoints
      torch.save(eff_model.state_dict(), os.path.join(DIR.CKPT_DIR, "last_model.pth"))

      # save best model and write logs to csv file
      if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(eff_model.state_dict(), os.path.join(DIR.CKPT_DIR, "best_model.pth"))
        with open(os.path.join(DIR.OUT_DIR, "best_logs.csv"), "a") as f:
          f.write(f"{epoch+1},{loss.item()},{mse.item()},{mde.item()},{fde.item()}\n")


      # Validation loop
      with torch.no_grad():
        for i, data in enumerate(val_loader):
          print("\r", end=f'{progress_bar(i, length=75, train_set_len=len(val_set), train_bs=config.VAL_BS)}')
          logging.info(f"Validation Epoch {epoch+1}, batch {i+1} / {len(val_loader)}")

          try: 
            val_loss, val_mse, val_mde, val_fde = train(data, train=False)
          except Exception as e:
            logging.error(e.__str__() + "Error in validation, in Line " + str(e.__traceback__.tb_lineno))

          try: 
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('MSE/val', val_mse, epoch)
            writer.add_scalar('MDE/val', val_mde, epoch)
            writer.add_scalar('FDE/val', val_fde, epoch)
          except Exception as e:
            logging.error(e.__str__() + "Error in writing to tensorboard, in Line " + str(e.__traceback__.tb_lineno))

          # if i >= 1:
          #   break

        print("Val Loss: {}".format(val_loss.item()))
    except Exception as e:
      logging.error(e.__str__() + f"in epoch {epoch+1}, in Line" + str(e.__traceback__.tb_lineno))
      continue

  prof.stop()
  writer.close()
  print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
