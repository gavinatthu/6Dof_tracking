import numpy as np
import os
import torch
from tqdm import tqdm

from data_loader.testing_6Dof import TestDatabase, TrainDatabase
from model.loss import compute_loss_snn_6Dof
from model.metric import medianRelativeError, rmse
from torch.utils.tensorboard import SummaryWriter   
from .gpu import moveToGPUDevice
from .tbase import TBase
from model.SNN_cnn_6Dof import *

class Tester(TBase):
    def __init__(self, data_dir, write, log_config, general_config):
        super().__init__(data_dir, log_config, general_config)
        self.write_output = write
        self.output_dir = self.log_config.getOutDir()

        test_database = TestDatabase(self.data_dir)
        self.test_loader = torch.utils.data.DataLoader(
                test_database,
                batch_size=general_config['batchsize'],
                shuffle=False,
                num_workers=general_config['hardware']['readerThreads'],
                pin_memory=True,
                drop_last=False)

        self.data_collector = DataCollector(general_config['simulation']['tStartLoss'])

    def test(self):
        self._loadNetFromCheckpoint()
        self.net = self.net.eval()
        #print(self.net)
        with torch.no_grad():
            for data in tqdm(self.test_loader, desc='testing'):
                data = moveToGPUDevice(data, self.device, self.dtype)

                spike_tensor = data['spike_tensor']     #torch.Size([24, 2, 180, 240, 100])
                ang_vel_gt = data['angular_velocity']   #torch.Size([24, 3, 100])

                ang_vel_pred = self.net(spike_tensor)   #torch.Size([24, 3, 100])
                print('Input shape:', spike_tensor.shape, 'Output shape:', ang_vel_pred.shape)
                self.data_collector.append(ang_vel_pred, ang_vel_gt, data['file_number'])
        if self.write_output:
            self.data_collector.writeToDisk(self.output_dir)
        self.data_collector.printErrors()

class Trainer(TBase):
    def __init__(self, data_dir, write, log_config, general_config):
        super().__init__(data_dir, log_config, general_config)

        self.time_win = 10
        self.write_output = write
        self.output_dir = self.log_config.getOutDir()
        test_database = TestDatabase(self.data_dir)
        train_database = TrainDatabase(self.data_dir)
        
        self.train_loader = torch.utils.data.DataLoader(
                train_database,
                batch_size=general_config['batchsize'],
                shuffle=True,
                num_workers=general_config['hardware']['readerThreads'],
                pin_memory=True,
                drop_last=False)

        self.test_loader = torch.utils.data.DataLoader(
                test_database,
                batch_size=general_config['batchsize'],
                shuffle=False,
                num_workers=general_config['hardware']['readerThreads'],
                pin_memory=True,
                drop_last=False)
        self.data_collector = DataCollector(general_config['simulation']['tStartLoss'])

    def train(self, num_epochs, learning_rate):
        self._trainfromscratch()    # train from scratch
        #self._loadNetFromCheckpoint()      # train from pretrained model
        self.net = self.net.train()

        act_fun = ActFun.apply
        self.hebb_tuple = self.net.produce_hebb()
        #print(self.output_dir)
        writer = SummaryWriter(self.output_dir)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            for i , data in enumerate(self.train_loader):
                self.net.zero_grad()
                optimizer.zero_grad()
                data = moveToGPUDevice(data, self.device, self.dtype)
                
                spike_tensor = data['spike_tensor']     #torch.Size([24, 2, 180, 240, 30])
                ang_vel_gt = data['angular_velocity']   #torch.Size([24, 3])

                ang_vel_pred, self.hebb_tuple = self.net(spike_tensor, self.hebb_tuple, self.time_win)   #torch.Size([24, 3])


                loss = compute_loss_snn_6Dof(ang_vel_pred, ang_vel_gt)

                loss.backward()
                optimizer.step()
                ang_vel_pred = ang_vel_pred.unsqueeze(2)
                ang_vel_pred = ang_vel_pred.repeat(1,1,self.time_win)    #torch.Size([24, 3, 30])
                #self.data_collector.append(ang_vel_pred, ang_vel_gt, data['file_number'])
                if (i+1) % 100 == 0:
                    print('Epoch: [{}/{}], Step: [{}/{}], Loss: {}'
                        .format(epoch+1, num_epochs, i+1, len(self.train_loader), loss.item()))
            
            writer.add_scalar('train loss', loss, global_step=epoch)
            self.test()
            writer.add_scalar('test loss', self.test_loss, global_step=epoch)
            
        #if self.write_output:
            #self.data_collector.writeToDisk(self.output_dir)
        #self.data_collector.printErrors()

    def test(self):
        self.net = self.net.eval()
        loss_list =[]

        with torch.no_grad():
            for i , data in enumerate(self.test_loader):
                data = moveToGPUDevice(data, self.device, self.dtype)

                spike_tensor = data['spike_tensor']
                ang_vel_gt = data['angular_velocity']

                ang_vel_pred , self.hebb_tuple = self.net(spike_tensor, self.hebb_tuple, self.time_win)
                
                loss = compute_loss_snn_6Dof(ang_vel_pred, ang_vel_gt)
                loss_list.append(loss)

                ang_vel_pred = ang_vel_pred.unsqueeze(2)
                ang_vel_pred = ang_vel_pred.repeat(1,1,self.time_win)    #torch.Size([24, 3, 100])
                
                #print('test loss:', loss)
                self.data_collector.append(ang_vel_pred, ang_vel_gt, data['file_number'])
        loss_list = torch.tensor(loss_list)
        
        self.test_loss = torch.mean(loss_list)
        print('test loss:', self.test_loss)
        if self.write_output:
            self.data_collector.writeToDisk(self.output_dir)
        self.data_collector.printErrors()

class DataCollector:
    def __init__(self, loss_start_idx: int):
        assert loss_start_idx >= 0
        self.loss_start_idx = loss_start_idx

        self.file_indices = list()
        self.data_gt = list()
        self.data_pred = list()

    def append(self, pred: torch.Tensor, gt: torch.Tensor, file_indices: torch.Tensor):
        # pred/gt: (batchsize, 3, time) tensor
        # file_indices: (batchsize) tensor

        self.data_pred.append(pred.detach().cpu().numpy())
        self.data_gt.append(gt.detach().cpu().numpy())
        self.file_indices.append(file_indices.to(torch.long).cpu().numpy())

    def writeToDisk(self, out_dir: str):
        pred = np.concatenate(self.data_pred, axis=0)
        gt = np.concatenate(self.data_gt, axis=0)
        file_indices = np.concatenate(self.file_indices)
        np.save(os.path.join(out_dir, 'predictions.npy'), pred)
        np.save(os.path.join(out_dir, 'groundtruth.npy'), gt)
        np.save(os.path.join(out_dir, 'indices.npy'), file_indices)

    def printErrors(self):
        pred = np.concatenate(self.data_pred, axis=0)
        gt = np.concatenate(self.data_gt, axis=0)
        pred = pred[..., self.loss_start_idx:]
        gt = gt[..., self.loss_start_idx:]
        
        print('RMSE: {} deg/s'.format(rmse(pred, gt, deg=True)))
        print('median of relative error: {}'.format(medianRelativeError(pred, gt)))
