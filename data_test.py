import numpy as np


PATH = "./logs/train/20210714-123633/out/"

gt = np.load(PATH+"groundtruth.npy")
gt_new = np.mean(gt,axis = 2)
#gt_new = gt[:,:,-1]
error = (gt[:,:,-1]-gt[:,:,0])
print(error[23313])
indices = np.load(PATH + "indices.npy")
pd = np.load(PATH + "predictions.npy")
pd_new = pd[:,:,0]

print(gt_new.shape)

print(gt_new[22315])
print(pd_new[22315])