"""
==============================
Filling the area between lines
==============================

This example shows how to use `~.axes.Axes.fill_between` to color the area
between two lines.
"""

import matplotlib.pyplot as plt
import numpy as np


def load(input_file):
    x, y_exp, y_base = [], [], []

    slot2nums = {}
    for data in open(input_file):
        parts = data.strip().split(' ')
        slot = ' '.join(parts[0].split('_')[0:1])
        exp_num, base_num = float(parts[1]), float(parts[2])
        if slot not in slot2nums:
            slot2nums[slot] = [exp_num, base_num]
        else:
            slot2nums[slot][0] += exp_num
            slot2nums[slot][1] += base_num

    slot2nums = sorted(slot2nums.items(), key=lambda x: x[0])
    print(slot2nums)

    labels = []
    for i, (slot, (exp, base)) in enumerate(slot2nums):
        x.append(i)
        y_exp.append(exp)
        y_base.append(base)
        labels.append(slot)
    x = np.array(x)
    y_exp = np.array(y_exp)
    y_base = np.array(y_base)

    return y_exp, y_base, slot2nums


fig, axs = plt.subplots(1, 4, figsize=(36, 10))

# fig.suptitle(r'Error Distribution of Ours$_\text{R}$ and Ours$_{\text{R&G}}$', fontsize=32)

input_files = [
    'error.1k.txt',
    'error.2k.txt',
    'error.4k.txt',
    'error.all.txt',
]
for i, ax in enumerate(axs):
    y_exp, y_base, slot2nums = load(input_files[i])

    species = [x[0] for x in slot2nums]
    penguin_means = {
        r'Ours$_\text{R}$': y_base,
        r'Ours$_\text{R&G}$': y_exp,
    }

    x = np.arange(len(species))  # the label locations
    width = 0.45  # the width of the bars
    multiplier = 0

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        # ax.bar_label(rects, padding=3, fontsize=16)
        multiplier += 1

    tag = input_files[i].split('.')[1].capitalize()
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(r'' + f'{tag} Real Data', fontsize=32)

    if i == 0:
        ax.set_ylabel('Number of errors', fontsize=32)
        ax.tick_params(axis='y', labelsize=32, labelrotation=90)
    else:
        ax.yaxis.set_ticklabels([])

    # ax.set_ylabel('Total Errors for Each Domain')
    # ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + width * 0.5, species, fontsize=24)
    ax.legend(loc='upper left', ncols=3, fontsize=28)
    ax.set_ylim(0, 2600)

axs[0].set_ylabel('Number of errors',fontsize=28)
fig.tight_layout()#调整整体空白
plt.subplots_adjust(wspace=0.025, hspace =0)#调整子图间距
plt.savefig('pic2.pdf', bbox_inches='tight')
# plt.show()


# fig, (ax) = plt.subplots(1, 1, sharex=True, figsize=(24, 8))
# ax.set_xlim(0, len(x) - 1)
# ax.set_ylim(0, max(max(y_exp), max(y_base)))
# ax.tick_params(axis='both', which='major', labelsize=24)  # 设置坐标标签大小为 12
# ax.set_title(r'Error Distribution of Ours$_\text{R}$ and Ours$_{\text{R&G}}$', fontsize=30)
# ax.plot(x, y_base, '-', color='C1')
# ax.plot(x, y_exp, '--', color='C0')
# ax.legend([r'Ours$_\text{R}$', r'Ours$_\text{R&G}$'], prop = {'size':24})
# ax.fill_between(x, y_exp, y_base, where=(y_exp > y_base), color='C0', alpha=0.3,
#                  interpolate=True)
# ax.fill_between(x, 0, y_base, where=(0 <= y_base), color='C1', alpha=0.3,
#                  interpolate=True)
# # plt.xticks(x, labels, rotation=90)
# plt.subplots_adjust(bottom=0)
# plt.xlim(-1, 100)
# plt.xlim(0, len(x) - 1)
# ax.set_xticks([])
# ax.set_xticklabels([])
#
# ax.set_xlabel('Different Types of Errors',fontsize=26)
# ax.set_ylabel('Number of errors',fontsize=26)
# plt.savefig('pic.pdf', bbox_inches='tight')