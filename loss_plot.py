#!/usr/bin/env python

# Used to create a plot for the report

xcpetion = [
    "2139f",
    "af619",
    "3dee6",
    "a5c30",
    "8a40b",
    "d57b6",
    "39fc1",
    "8c755",
    "eaaa8",
    "ce281",
    "f2915",
    "a0ad0",
    "c3f90",
    "9268e",
    "a9ccf",
    "c7dea",
    "ab129",
    "ec02b",
    "38630",
    "d372f",
    "39c6e",
    "51e8b",
    "dd8a3",
    "54e63",
    "cb318",
    "48ecf",
    "c30e4",
    "097bc",
    "3ddf4",
    "a19aa",
    "b98a5",
    "5564a",
    "7f1e3",
    "4959c",

]

efficientnet = [
    "6515d",
    "6832c",
    "a2acb",
    "6b390",
    "4d54b",
    "c0cd1",
    "c4c51",
    "8da84",
    "ffad4",
    "324d5",
    "7199d",
    "826fd",
    "62e70",
    "c6527",
    "04828",
    "d4e43",
    "a83b6",
    "cc85e",
    "c6384",
    "3d288",
    "de401",
    "80fab",
    "27098",
    "8663f",
    "35840",
    "7c971",
    "72cb5",
    "8c95d",
    "11333",
    "0a7e8",
    "ddf70",
    "f8715",
    "aeed5",
    "badb7",
    "8ca89",
    "9722f",
    "f1765",
    "93181",
    "33519",
    "46651",
    "43d5f",
    "c2d6b",
    "a1320",
    "21031",
    "19b90",
    "1473d",
    "3cc09",
    "8b4a5",
    "19df2",
    "70795",
    "22cdc",
    "23a4d",
    "837803d5b2a343b19765215215f21f47",
    "6164403074e541c29c221ff43c28f39a",
    "ae2770a1eb3f42bb9e591c835e114667"
]

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data_pd = pd.DataFrame(columns=['Epoch', 'Model', 'Run' 'Loss'], dtype=int)

# Opening JSON file
f = open('data.json', )

# returns JSON object as
# a dictionary
data = json.load(f)

# Iterating through the json
# list
xcpetion_counter = 0
efficientnet_counter = 0

for i in data['data']:

    if i['name'] in xcpetion:
        for j in range(len(i['x'])):
            data_pd = data_pd.append({'Epoch': j, 'Model': "Xception", 'Run': xcpetion_counter, 'Loss': i['y'][j]}, ignore_index=True)
        xcpetion_counter += 1

    if i['name'] in efficientnet:
        for j in range(len(i['x'])):
            data_pd = data_pd.append({'Epoch': j, 'Model': "EfficientNet", 'Run': efficientnet_counter, 'Loss': i['y'][j]}, ignore_index=True)
        efficientnet_counter += 1

print(data_pd)
# Closing file

f.close()

sns.set(style="whitegrid", rc={'figure.figsize': (16, 9)}, font_scale=2)

ax = sns.lineplot(data=data_pd, x="Epoch", y="Loss", hue='Model', style='Run', dashes=False)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:3], labels=labels[1:3])

# ax.set(title='Convergence')


# plt.show()

plt.savefig("convergence_roadsegment.pdf", bbox_inches="tight")
