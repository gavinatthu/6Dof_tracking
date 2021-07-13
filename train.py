import numpy as np
import argparse
import os
# Must be set before importing torch.
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from config.utils import getTestConfigs
#from utils import Tester
from utils.testing_6Dof import Tester, Trainer


parser = argparse.ArgumentParser(description='Train the SNN model')
parser.add_argument('--datadir',
                    #default=os.path.join(os.getcwd(), 'data'),
                    default=os.path.abspath('/data1/DVSAngular/'),
                    help='Data directory')
parser.add_argument('--logdir',
                    default=os.path.join(os.getcwd(), 'logs/train'),
                    help='Test logging directory')
parser.add_argument('--config',
                    default=os.path.join(os.getcwd(), 'train_config.yaml'),
                    help='Path to test config file')
parser.add_argument('--write',
                    action='store_false',
                    help='Write network predictions to logging sub-directory')

args = parser.parse_args()
configs = getTestConfigs(args.logdir, args.config)

trainer = Trainer(args.datadir, args.write, configs['log'], configs['general'])

trainer.train(num_epochs=200, learning_rate=0.0001)  # LR 不能设置更大 否则无法收敛
trainer.test()
