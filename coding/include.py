import os
from datetime import datetime
# PROJECT_PATH = os.path.dirname(os.path.realpath(__file__).replace('/coding/20190910/code/dummy_11a/lib',''))
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__).replace('/lib',''))
IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


#numerical libs
import math
import numpy as np
import random
import PIL
import cv2
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('WXAgg')
#matplotlib.use('Qt4Agg')
#matplotlib.use('Qt5Agg') #Qt4Agg
print('matplotlib.get_backend : ', matplotlib.get_backend())
#print(matplotlib.__version__)


# torch libs
import torch

from torch.utils.data.sampler import *
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel

from torch.nn.utils.rnn import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


# std libs
import collections
import copy
import numbers
import inspect
import shutil
from timeit import default_timer as timer
import itertools
from collections import OrderedDict
from multiprocessing import Pool
import multiprocessing as mp

#from pprintpp import pprint, pformat
import json
import zipfile
from math import sin, cos




import csv
import pandas as pd
import pickle
import glob
import sys
from distutils.dir_util import copy_tree
import time
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from functools import reduce
from pathlib import Path
import struct
import sklearn.metrics
from functools import partial
from sklearn.model_selection import train_test_split




import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from PIL import Image


from mpl_toolkits.mplot3d import Axes3D



# constant #
PI  = np.pi
INF = np.inf
EPS = 1e-12





