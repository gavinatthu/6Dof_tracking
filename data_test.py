import numpy as np
import h5py
import os

#TODO 画图观察实际数据表现
def result_test():
    PATH = "./logs/train/20210714-171404/out/"

    gt = np.load(PATH+"groundtruth.npy")
    gt_new = np.mean(gt,axis = 2)
    #gt_new = gt[:,:,-1]
    error = (gt[:,:,-1]-gt[:,:,0])

    indices = np.load(PATH + "indices.npy")
    pd = np.load(PATH + "predictions.npy")
    pd_new = pd[:,:,0]

    print(gt_new.shape)

    print(gt_new[23314])
    print(pd_new[23314])


def printname(name):
     print(name)


def DSEC():
    PATH = "/data1/DSEC/train_events/interlaken_00_c/events/left/events.h5"
    f = h5py.File(PATH, "r")
    f.visit(printname)
    print(f['events']['t'])

def MVSEC():
    PATH = "/data1/MVSEC/outdoor_day1_data.hdf5"
    f = h5py.File(PATH, "r")
    f.visit(printname)
    print(f['davis']['left']['events'])


if __name__=='__main__':

    MVSEC()