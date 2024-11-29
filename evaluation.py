import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
import copy
import time
import numpy as np
import random

def plot_box(GT, result):
    b = np.where(GT == 1)
    c = np.where(GT == 0)
    pos = result[b]
    neg = result[c]
    plt.boxplot((pos, neg), labels=('pos','neg'))
    plt.show()

def plot_ROC(test_labels, resultall, name, image_name,show=True):
    mark_list = ['o', 'v', 'o', 'v','o', 'v', '*', 'x', 'D', 's', 'P', 'h','.', ',', ]
    mark_size = [1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1,2, 2 ]
    if show:
        plt.subplots(num='ROC curve', figsize = [6,4], dpi=100)
    auc_list = []
    ft_list = []
    for i in range(len(resultall)):
        fpr, tpr, thresholds = metrics.roc_curve(
         test_labels, resultall[i], pos_label=1)  # caculate False alarm rate and Probability of detection
        auc = "%.5f" % metrics.auc(fpr, tpr)     # caculate AUC (Area Under the Curve)
        ft = "%.5f" % metrics.auc(thresholds, fpr)
        auc_list.append(float(auc))
        ft_list.append(float(ft))
        print('%s_AUC: %s'%(name[i],auc))
        print('%s_FT: %s' % (name[i], ft))
        if show:
            if not i: my_plot = plt.semilogx if metrics.auc(fpr, tpr) > 0.9 else plt.plot
            my_plot(fpr, tpr, label=name[i],marker = mark_list[i],markersize = mark_size[i] + 0.5, lw=1)
    if show:
        plt.xlim([1e-5, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right', facecolor='none', edgecolor='none')
        plt.title('ROC Curve of ' + image_name.replace('ablation_study',''))
        if os.path.exists(image_name + '.png'):
            os.remove(image_name + '.png')
        plt.savefig(image_name + '.png')
        plt.show()
    # show ROC curve
    return auc_list, ft_list