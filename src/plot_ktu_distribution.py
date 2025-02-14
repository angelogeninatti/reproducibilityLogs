import json 
import numpy as np 
import matplotlib.pyplot as plt

with open('./results/rankings.json') as f_in:
    rankings = json.loads(f_in.read())

B = [round(0.2 + 0.05 * i, 3) for i in range(0,13)]

data = {}
for qid, _rankings in rankings.items():
    ktu = [_rankings.get('(10, {}, 0.8)'.format(str(b))).get('ktu') for b in B]
    ktu_set = set(ktu)
    d = {}
    for k in ktu_set:
        if k != 1.0:
            idx = ktu.index(k)
            _b = str(round(0.2 + 0.05 * idx, 3))
            params = '(10, {}, 0.8)'.format(str(_b))
            d[k] = params
    d[1.0] = '(10, 0.5, 0.8)'
    data[qid] = d

# Determine global x-axis limits
all_keys = np.hstack([list(inner_dict.keys()) for inner_dict in data.values()])
min_x, max_x = min(all_keys), max(all_keys)

# Set up the figure and axes for plotting
fig, ax = plt.subplots(int(len(data) / 4), 4, figsize=(12, .25 * len(data)))

# If there's only one plot, ax might not be an array which could raise an error, so we ensure it's always an array
if len(data) == 1:
    ax = np.array([ax])

for i, (key, inner_dict) in enumerate(data.items()):
    keys = list(inner_dict.keys())
    values = [1] * len(keys)  # Placeholder for the y-axis values. Adjust as per your actual values.

    ax[i - (i // int(len(data)/4)) * int(len(data) / 4)][i // int(len(data)/4)].bar(keys, values)
    ax[i - (i // int(len(data)/4)) * int(len(data) / 4)][i // int(len(data)/4)].set_title(f"Query {key}")
    # ax[i - (i // int(len(data)/2)) * int(len(data) / 2)][i // int(len(data)/2)].set_xlabel("Key values")
    ax[i - (i // int(len(data)/4)) * int(len(data) / 4)][i // int(len(data)/4)].set_xlim(min_x, max_x)

plt.tight_layout()
plt.savefig('./figures/ktu_distribution_bs_10_k1_0.8.pdf', format='pdf', bbox_inches='tight')


data = {}
for qid, _rankings in rankings.items():
    ktu = [_rankings.get('({}, 0.8)'.format(str(b))).get('ktu') for b in B]
    ktu_set = set(ktu)
    d = {}
    for k in ktu_set:
        if k != 1.0:
            idx = ktu.index(k)
            _b = str(round(0.2 + 0.05 * idx, 3))
            params = '({}, 0.8)'.format(str(_b))
            d[k] = params
    d[1.0] = '(0.5, 0.8)'
    data[qid] = d

# Determine global x-axis limits
all_keys = np.hstack([list(inner_dict.keys()) for inner_dict in data.values()])
min_x, max_x = min(all_keys), max(all_keys)

# Set up the figure and axes for plotting
fig, ax = plt.subplots(int(len(data) / 4), 4, figsize=(12, .25 * len(data)))

# If there's only one plot, ax might not be an array which could raise an error, so we ensure it's always an array
if len(data) == 1:
    ax = np.array([ax])

for i, (key, inner_dict) in enumerate(data.items()):
    keys = list(inner_dict.keys())
    values = [1] * len(keys)  # Placeholder for the y-axis values. Adjust as per your actual values.

    ax[i - (i // int(len(data)/4)) * int(len(data) / 4)][i // int(len(data)/4)].bar(keys, values)
    ax[i - (i // int(len(data)/4)) * int(len(data) / 4)][i // int(len(data)/4)].set_title(f"Query {key}")
    # ax[i - (i // int(len(data)/2)) * int(len(data) / 2)][i // int(len(data)/2)].set_xlabel("Key values")
    ax[i - (i // int(len(data)/4)) * int(len(data) / 4)][i // int(len(data)/4)].set_xlim(min_x, max_x)

plt.tight_layout()
plt.savefig('./figures/ktu_distribution_k1_0.8.pdf', format='pdf', bbox_inches='tight')
