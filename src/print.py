import os
import copy
import time
import pickle
import numpy as np
import random
from gan import GeneratorA
from tqdm import tqdm

import torch

# path_project = '/home/aiia611/wqb/data'
# model_path = os.path.join('{}/distillation_checkpoints'.format(path_project), 'training_checkpoint_cifar_1_iid[0]_client[9].pth.tar')
# assert os.path.isfile(model_path)
# checkpoint = torch.load(model_path)
# print(checkpoint.keys())
# print(checkpoint['test_acc'])
a = [1,2,3,4,5]
for i in range(len(a)):
    a[i] = a[i] * 50

print(a)