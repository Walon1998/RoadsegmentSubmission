#!/usr/bin/env python


import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

google_data = [
    # ["Logistic Regression", 0.50971],
    ["CNN on patches", "Baseline", 0.84150],
    ["U-Net", "Baseline", 0.90998],
    # ["GC-DCNN", "Baseline", 0.89276],

    ["Xception 128", "Xception", 0.91015],
    ["Xception 256", "Xception", 0.92097],
    ["Xception 400", "Xception", 0.92593],
    ["Xception Pretrained", "Xception", 0.91722],
    ["EfficientNet 128", "EfficientNet", 0.90288],
    ["EfficientNet 256", "EfficientNet", 0.92282],
    ["EfficientNet 400", "EfficientNet", 0.92671],
    ["EfficientNet Pretrained", "EfficientNet", 0.92039],

    ["Top 3 Xception", "Ensemble", 0.92805],
    ["Top 3 EfficientNet", "Ensemble", 0.92853],

    ["Top 3 Ensemble", "Ensemble", 0.92897],
    ["Top 5 Ensemble", "Ensemble", 0.92972],
    ["Top 10 Ensemble", "Ensemble", 0.93002],
    ["Top 20 Ensemble", "Ensemble", 0.93094],

]

orig_data = [
    # ["Logistic Regression", 0.50971],
    ["CNN on patches", "Baseline", 0.79537],
    ["U-Net", "Baseline", 0.85905],
    # ["GC-DCNN", "Baseline", 0.82951],
    ["Xception 400", "Xception", 0.87608],
    ["Xception Pretrained", "Xception", 0.85746],
    ["EfficientNet 400", "EfficientNet", 0.86044],
    ["EfficientNet Pretrained", "EfficientNet", 0.87522],

]

# patched_data = [
#     ["Xception 400", "Xception", 0.85999],
#     ["EfficientNet 400", "EfficientNet", 0.89738],
#
# ]

data_pd = pd.DataFrame(columns=['Score', 'Model', 'Dataset'], dtype=int)

for entry in google_data:
    data_pd = data_pd.append({'Score': entry[2], 'Model': entry[0], 'Dataset': 'Google', 'Family': entry[1]}, ignore_index=True)

for entry in orig_data:
    data_pd = data_pd.append({'Score': entry[2], 'Model': entry[0], 'Dataset': 'Kaggle', 'Family': entry[1], }, ignore_index=True)

# for entry in patched_data:
#     data_pd = data_pd.append({'Score': entry[2], 'Model': entry[0], 'Dataset': 'Patched', 'Family': entry[1], }, ignore_index=True)

sns.set(style="white", rc={'figure.figsize': (16, 9)}, font_scale=1.5)

fig, ax = plt.subplots(4, 1, tight_layout=True, figsize=(16, 7.5), sharex=True, sharey=False)

palette_0 = sns.color_palette()

size = 1000
alpha = 1

sns.scatterplot(data=data_pd[data_pd['Family'] == "Baseline"], x="Score", y="Dataset", hue='Model', s=size, alpha=alpha, marker="o", ax=ax[0])
sns.scatterplot(data=data_pd[data_pd['Family'] == "Xception"], x="Score", y="Dataset", hue='Model', s=size, alpha=alpha, marker="X", ax=ax[1])
sns.scatterplot(data=data_pd[data_pd['Family'] == "EfficientNet"], x="Score", y="Dataset", hue='Model', s=size, alpha=alpha, marker="^", ax=ax[2])
sns.scatterplot(data=data_pd[data_pd['Family'] == "Ensemble"], x="Score", y="Dataset", hue='Model', s=size, alpha=alpha, marker="|", ax=ax[3])

handles0, labels0 = ax[0].get_legend_handles_labels()
handles1, labels1 = ax[1].get_legend_handles_labels()
handles2, labels2 = ax[2].get_legend_handles_labels()
handles3, labels3 = ax[3].get_legend_handles_labels()

handles = []
labels = labels0 + labels1 + labels2 + labels3

for i in range(len(handles0)):
    handles.append(mlines.Line2D([0], [0], markersize=15, linestyle="none", marker="o", color=palette_0[i]))

for i in range(len(handles1)):
    handles.append(mlines.Line2D([0], [0], markersize=15, linestyle="none", marker="X", color=palette_0[i]))

for i in range(len(handles2)):
    handles.append(mlines.Line2D([0], [0], markersize=15, linestyle="none", marker="^", color=palette_0[i]))

for i in range(len(handles3)):
    handles.append(mlines.Line2D([0], [0], markersize=15, linestyle="none", marker="|", color=palette_0[i]))

ax[0].legend().remove()
ax[1].legend().remove()
ax[2].legend().remove()
ax[3].legend().remove()

ax[0].grid(axis='x')
ax[1].grid(axis='x')
ax[2].grid(axis='x')
ax[3].grid(axis='x')

ax[0].set_ylim(-0.5, 1.5)
ax[1].set_ylim(-0.5, 1.5)
ax[2].set_ylim(-0.5, 1.5)
ax[3].set_ylim(-0.5, 1.5)

fig.legend(handles=handles, labels=labels, bbox_to_anchor=(1.24, 1.0), fontsize=20)

# plt.show()

plt.savefig("summary_roadsegment.pdf", bbox_inches="tight")
