import numpy as np


PATH = "./logs/train/20210713-161833/out/"

gt = np.load(PATH+"groundtruth.npy")
gt_new = np.mean(gt,axis = 2)


indices = np.load(PATH + "indices.npy")
pd = np.load(PATH + "predictions.npy")
pd_new = pd[:,:,0]

print(gt_new.shape)

print(gt_new[-1])
print(pd_new[-1])