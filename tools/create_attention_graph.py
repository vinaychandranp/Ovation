import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import argparse
import numpy as np
from matplotlib import colors
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--path', type=str, help='Path to save files with results from attention model')
# args = parser.parse_args()
#
# #PARAMETER
# id = 120
#
#
# review = np.squeeze(np.load(args.path + '_review.npy'))
# score = np.squeeze(np.load(args.path + '_score.npy'))
# attention = np.squeeze(np.load(args.path + '_attention.npy'))
# gt = np.squeeze(np.load(args.path + '_gt.npy'))
#
# length = len(review[id].split(' '))
# attention = np.array([item[:length]/np.sum(item[:length])*100 for item in attention])

# attention = np.array([[i/np.sum(i) * 100 for i in item] for item in attention])

def plot_attention(review, attention, length):
    fig, ax = plt.subplots(figsize=(10,5))
    heatmap = ax.pcolor(attention[:,:-1,:length], cmap=plt.cm.Blues, alpha=0.9)
    X_label = review[id].split(' ')
    xticks = range(0,len(X_label))
    ax.set_xticks(xticks, minor=False) # major ticks
    ax.set_xticklabels(X_label, minor = False, rotation=45)   # labels should be 'unicode'
    ax.grid(True)
    fig.tight_layout()
    return fig
# plt.savefig('/tmp/tmp.pdf', bbox_inches='tight')
