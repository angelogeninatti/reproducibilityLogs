import json 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ir_datasets

with open('./results/rankings.json', 'r') as f_in:
    rankings = json.loads(f_in.read())

selected_queries_only = True
exclude = ['2033470', '2025747', '2055795', '2046371', '2002533', '2032956', '2003322', '2038890', '2034676', '2005861', '2006211', '2006627', '2007055', '2007419', '2008871', '2032090', '2049687', '2017299']
selection = []
dataset = ir_datasets.load("msmarco-passage-v2/trec-dl-2022/judged")
for query in dataset.queries_iter():
    if query.query_id not in exclude:
        selection.append(query.query_id)
rankings = {key: rankings[key] for key in selection if key in rankings}

bs = [4, 10, 50]
B = [round(0.2 + 0.05 * i, 3) for i in range(0,13)]
K1 = [round(0.5 + 0.05 * i, 3) for i in range(0,13)]

systems = []

for b in B:
    for k1 in K1:
        systems.append('({}, {})'.format(b, k1))

for _bs in bs:
    for b in B:
        for k1 in K1:
            systems.append('({}, {}, {})'.format(_bs, b, k1))

# for system in systems:
#     # Create figure
#     plt.figure(figsize=(15, 5))
#     # Setup GridSpec layout
#     gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[3, 1])  # Example: 3:1 width ratio
#     # Creating subfigures
#     ax0 = plt.subplot(gs[0])
#     ax1 = plt.subplot(gs[1])
#     scores = [rankings.get(qid).get(system).get('ktu') for qid in rankings.keys()]
#     qids = rankings.keys()
#     sorted_pairs = sorted(zip(qids, scores), key=lambda x: x[1], reverse=True)
#     sorted_labels, sorted_values = zip(*sorted_pairs)
#     # ax0.figure(figsize=(12,3))
#     ax0.bar(sorted_labels, sorted_values, color='skyblue', edgecolor='black') 
#     ax0.set_title('{} [Avg. KTU ={:3f}]'.format(system, float(sum(scores) / len(scores))))
#     ax0.set_xlabel('Query identifier')
#     ax0.set_ylabel('KTU scores')
#     ax0.set_xticklabels(list(sorted_labels), rotation='vertical')
#     ax1.hist(scores, bins=25, color='salmon', edgecolor='black')
#     ax1.set_title('Histogram')
#     ax1.set_xlabel('KTU scores')
#     ax1.set_ylabel('Frequency')
#     plt.tight_layout() 
#     plt.savefig('./results/score_distributions/ktu/{}.pdf'.format(system))
#     # plt.show()

for system in systems:
    # Create figure
    plt.figure(figsize=(15, 5))
    # Setup GridSpec layout
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[3, 1])  # Example: 3:1 width ratio
    # Creating subfigures
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    scores = [rankings.get(qid).get(system).get('ndcg_cut_5') for qid in rankings.keys()]
    qids = rankings.keys()
    sorted_pairs = sorted(zip(qids, scores), key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_values = zip(*sorted_pairs)
    # ax0.figure(figsize=(12,3))
    ax0.bar(sorted_labels, sorted_values, color='skyblue', edgecolor='black') 
    ax0.set_title('{} [Avg. nDCG@5 ={:3f}]'.format(system, float(sum(scores) / len(scores))))
    ax0.set_xlabel('Query identifier')
    ax0.set_ylabel('nDCG@5 scores')
    ax0.set_xticklabels(list(sorted_labels), rotation='vertical')
    ax1.hist(scores, bins=25, color='salmon', edgecolor='black')
    ax1.set_title('Histogram')
    ax1.set_xlabel('nDCG scores')
    ax1.set_ylabel('Frequency')
    plt.tight_layout() 
    plt.savefig('./results/score_distributions/ndcg/{}.pdf'.format(system))
    # plt.show()

# for system in systems:
#     # Create figure
#     plt.figure(figsize=(15, 5))
#     # Setup GridSpec layout
#     gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[3, 1])  # Example: 3:1 width ratio
#     # Creating subfigures
#     ax0 = plt.subplot(gs[0])
#     ax1 = plt.subplot(gs[1])
#     scores = [rankings.get(qid).get(system).get('P_5') for qid in rankings.keys()]
#     qids = rankings.keys()
#     sorted_pairs = sorted(zip(qids, scores), key=lambda x: x[1], reverse=True)
#     sorted_labels, sorted_values = zip(*sorted_pairs)
#     # ax0.figure(figsize=(12,3))
#     ax0.bar(sorted_labels, sorted_values, color='skyblue', edgecolor='black') 
#     ax0.set_title('{} [Avg. P@5 ={:3f}]'.format(system, float(sum(scores) / len(scores))))
#     ax0.set_xlabel('Query identifier')
#     ax0.set_ylabel('P@5 scores')
#     ax0.set_xticklabels(list(sorted_labels), rotation='vertical')
#     ax1.hist(scores, bins=25, color='salmon', edgecolor='black')
#     ax1.set_title('Histogram')
#     ax1.set_xlabel('P@5 scores')
#     ax1.set_ylabel('Frequency')
#     plt.tight_layout() 
#     plt.savefig('./results/score_distributions/precision/{}.pdf'.format(system))