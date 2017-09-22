import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import argparse
import numpy as np
from matplotlib import colors

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='Path to save files with results from attention model')
args = parser.parse_args()

review = np.squeeze(np.load(args.path + '_review.npy'))
score = np.squeeze(np.load(args.path + '_score.npy'))
attention = np.squeeze(np.load(args.path + '_attention.npy'))
gt = np.squeeze(np.load(args.path + '_gt.npy'))

cmap = colors.ListedColormap(['red', 'blue'])




