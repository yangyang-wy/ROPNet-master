from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
import os

ROOT = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(ROOT)))

from data import ModelNet40
from models import ROPNet
from loss import cal_loss
from metrics import compute_metrics, summary_metrics, print_train_info
from utils import time_calc, inv_R_t, batch_transform, setup_seed, square_dists
from configs import train_config_params as config_params
os.environ["CUDA_VISIBLE_DEVICES"] ="0"