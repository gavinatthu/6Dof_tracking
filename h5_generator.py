import h5py
import os
import numpy as np

#TODO: 分离测试、训练集，转换成多个样本的形式 timewin=30切分

PATH = "/data1/DVSAngular/shapes/"
time_win = 30
'''

events_data = np.loadtxt(PATH + "events.txt") # (17962477, 4)(timestamp x y polarity)
gt_data = np.loadtxt(PATH + "groundtruth.txt") #(11862, 8) (timestamp px py pz qx qy qz qw)
tmp = np.arange(len(events_data))
start_time = 0

for i in range(int(len(gt_data)/time_win)):

    index = tmp[(events_data[:,0]<gt_data[(i+1)*time_win,0])&(events_data[:,0]>=gt_data[i*time_win,0])]
    print(i, len(index))
    f = h5py.File(PATH+"train/Seq_"+str(i)+".h5", "w")
    f.create_dataset('ev_xy', data=events_data[index,1:3])
    f.create_dataset('ev_ts', data=1000*(events_data[index,0]-events_data[index[0],0]))
    f.create_dataset('ev_pol', data=events_data[index,3])

    f.create_dataset('ang_ts', data=1000*(gt_data[i*time_win:(i+1)*time_win,0]-gt_data[i*time_win,0]))
    f.create_dataset('ang_xyz', data=gt_data[i*time_win:(i+1)*time_win,1:])
    f.close()
'''


f = h5py.File("/data1/DVSAngular/shapes/train/Seq_0.h5", "r")
ev_xy = f['ev_xy'][()]
ev_ts = f['ev_ts'][()]
ev_pol = f['ev_pol'][()]
ang_ts = f['ang_ts'][()]
ang_xyz = f['ang_xyz'][()]
print(ev_ts, len(ang_ts))
f.close()
