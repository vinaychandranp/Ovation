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

cmap = matplotlib.cm.get_cmap('Spectral')
norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
fig, ax = plt.subplots()

attention = np.array([item[:10]/np.sum(item[:10])*100 for item in attention])

ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
ax.set_xticks(np.arange(10, len(review[0].split(' ')[:10]), 1))
ax.set_yticks(np.arange(1, attention.shape[1], 1));

plt.imshow(attention[0,:,:len(review[0].split(' ')[:10])], norm=norm, cmap=cmap)

plt.savefig('/tmp/tmp.pdf')


