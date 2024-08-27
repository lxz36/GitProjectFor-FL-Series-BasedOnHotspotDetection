#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


local_acc = np.asarray([0.92976, 0.93020, 0.92984, 0.92952, 0.92980])
centralized_acc = 0.97019
federated_acc = 0.88 # verify

y = np.array([centralized_acc, local_acc.mean(), federated_acc])
y_err = np.asarray([0, local_acc.ptp()*5, 0])
y_err = None
x = np.arange(y.shape[0])
x_labels = ['Centralized', 'Local', 'Federated']

fig, axs = plt.subplots(figsize=(3.8, 2.))

print(y_err)
axs.bar(x, y, yerr=y_err, align='center', ecolor='black', capsize=5,
        color=['r', 'g', 'b'])
axs.set_xticks(x)
axs.set_xticklabels(x_labels)
axs.set_ylim([.85, 1.])
axs.set_ylabel('Accuracy')
fig.tight_layout()
fig.savefig('toy_example.png', dpi=300)
