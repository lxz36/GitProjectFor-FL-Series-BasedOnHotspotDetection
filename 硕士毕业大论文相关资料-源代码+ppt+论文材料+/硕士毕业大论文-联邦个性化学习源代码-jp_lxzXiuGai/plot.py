#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

os.environ['OMP_DISPLAY_ENV'] = 'FALSE'


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{0:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


figsize = np.array([4, 3]) * 1.3    # default = 1.6

group_norms = np.load('group_norm-lasso.npy')

print('Found {} rounds'.format(group_norms.shape[0]))

fig, axs = plt.subplots(figsize=figsize)
axs.bar(np.arange(group_norms.shape[1]), group_norms[-1])
fig.savefig('group_norm-lasso.png', dpi=300)
plt.close(fig)

sorted_idx = group_norms[-1].argsort()
print(sorted_idx[3:])

n_ft = [32, 29, 26, 23, 20]
acc_iccad = [0.89853, 0.89443, 0.90552, 0.89526, 0.86279]
acc_asml1 = [0.88625, 0.88268, 0.91085, 0.89070, 0.88421]

tpr_iccad = [0.97662, 0.96672, 0.97266, 0.97068, 0.96751]
tpr_asml1 = [0.83878, 0.83534, 0.87667, 0.84619, 0.83715]

fpr_iccad = [0.10289, 0.10688, 0.09570, 0.10611, 0.13912]
fpr_asml1 = [0.00936, 0.01320, 0.01397, 0.01141, 0.01231]


x = np.arange(len(n_ft))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=figsize)
rects1 = ax.bar(x - width/2, acc_iccad, width, label='ICCAD-2012')
rects2 = ax.bar(x + width/2, acc_asml1, width, label='ASML-1')

ax.set_ylabel('Accuracy')
ax.set_ylim([0.7, 1.01])
ax.set_title('Accuracy when applied selected features.')
ax.set_xticks(x)
ax.set_xlabel('#features')
ax.set_xticklabels(n_ft)
ax.legend(loc='lower right')

autolabel(ax, rects1)
autolabel(ax, rects2)

fig.tight_layout()
fig.savefig('acc.png', dpi=300)

fig, ax = plt.subplots(figsize=figsize)
rects1 = ax.bar(x - width/2, tpr_iccad, width, label='ICCAD-2012')
rects2 = ax.bar(x + width/2, tpr_asml1, width, label='ASML-1')

ax.set_ylabel('True Positive Rate')
ax.set_ylim([0.7, 1.01])
ax.set_title('True Positive Rate when applied selected features.')
ax.set_xticks(x)
ax.set_xlabel('#features')
ax.set_xticklabels(n_ft)
ax.legend(loc='lower right')

autolabel(ax, rects1)
autolabel(ax, rects2)

fig.tight_layout()
fig.savefig('tpr.png', dpi=300)


fig, ax = plt.subplots(figsize=figsize)
rects1 = ax.bar(x - width/2, fpr_iccad, width, label='ICCAD-2012')
rects2 = ax.bar(x + width/2, fpr_asml1, width, label='ASML-1')

ax.set_ylabel('False Positive Rate')
ax.set_ylim([0., .15])
ax.set_title('False Positive Rate when applied selected features.')
ax.set_xticks(x)
ax.set_xlabel('#features')
ax.set_xticklabels(n_ft)
ax.legend(loc='lower right')

autolabel(ax, rects1)
autolabel(ax, rects2)

fig.tight_layout()
fig.savefig('fpr.png', dpi=300)
