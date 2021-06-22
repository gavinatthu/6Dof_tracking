import h5py
import os
import shutil
import numpy as np
from random import sample

#TODO: 分离测试、训练集，转换成多个样本的形式 timewin=30切分

PATH = "/data1/DVSAngular/shapes/"
time_win = 30


events_data = np.loadtxt(PATH + "events.txt") # (17962477, 4)(timestamp x y polarity)
gt_data = np.loadtxt(PATH + "groundtruth.txt") #(11862, 8) (timestamp px py pz qx qy qz qw)
tmp = np.arange(len(events_data))
start_time = 0

for i in range(int(len(gt_data)/time_win)):

    index = tmp[(events_data[:,0]<gt_data[(i+1)*time_win,0])&(events_data[:,0]>=gt_data[i*time_win,0])]
    print('Seq_',i, 'Length ', len(index))
    f = h5py.File(PATH+"train/Seq_"+str(i)+".h5", "w")
    f.create_dataset('ev_xy', data=events_data[index,1:3])
    f.create_dataset('ev_ts', data=1000*(events_data[index,0]-gt_data[i*time_win,0]))
    print(f['ev_ts'][-1] - f['ev_ts'][0])
    f.create_dataset('ev_pol', data=events_data[index,3])

    f.create_dataset('ang_ts', data=1000*(gt_data[i*time_win:(i+1)*time_win,0]-gt_data[i*time_win,0]))
    f.create_dataset('ang_xyz', data=gt_data[i*time_win:(i+1)*time_win,1:])
    f.close()


# split test & training set ratio = 1/10
documents = list(os.walk(PATH + 'train'))[0]

test_name = sample(documents[2][1:], int(len(documents[2])/10))

for name in test_name:

    shutil.move(documents[0] + '/' + name, PATH+'test')
