import h5py
import os
import numpy as np

PATH = "/data1/DVSAngular/shapes/"
'''
events_data = np.loadtxt(PATH + "events.txt") # (17962477, 4)(timestamp x y polarity)
gt_data = np.loadtxt(PATH + "groundtruth.txt") #(11862, 8) (timestamp px py pz qx qy qz qw)

f = h5py.File(PATH+"train.h5", "w")
f.create_dataset('ev_xy', data=events_data[:,1:3])
f.create_dataset('ev_ts', data=events_data[:,0])
f.create_dataset('ev_pol', data=events_data[:,3])

f.create_dataset('ang_ts', data=gt_data[:,0])
f.create_dataset('ang_xyz', data=gt_data[:,4:])
f.create_dataset('vel_xyz', data=gt_data[:,1:4])


f.close()
'''
f = h5py.File(PATH+"train.h5", "r")
print(f['ev_xy'][()].shape)
