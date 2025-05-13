import json
import pickle
from pprint import pprint

import ir_datasets

with open('./results/rankings.json', 'r') as f_in:
    rankings = json.loads(f_in.read())

exclude = ['2033470', '2025747', '2055795', '2046371', '2002533', '2032956', '2003322', '2038890', '2034676', '2005861', '2006211', '2006627', '2007055', '2007419', '2008871', '2032090', '2049687', '2017299']
selection = []
dataset = ir_datasets.load("msmarco-passage-v2/trec-dl-2022/judged")
for query in dataset.queries_iter():
    if query.query_id not in exclude:
        selection.append(query.query_id)
rankings = {key: rankings[key] for key in selection if key in rankings}

bs = [4]
B = [round(0.2 + 0.05 * i, 3) for i in range(0,13)]
K1 = [round(0.5 + 0.05 * i, 3) for i in range(0,13)]

# Create output dictionary
output_dict = {}

for _bs in bs:
    for b in B:
        for k1 in K1:
            system = f"({_bs}, {b}, {k1})"
            for qid in rankings.keys():
                if qid not in output_dict:
                    output_dict[qid] = {}
                output_dict[qid][system] = rankings[qid][system]['ktu']

# Save to pkl
pprint(output_dict)
print(len(output_dict.keys()))
with open('./results/ktu.pkl', 'wb') as f_out:
    pickle.dump(output_dict, f_out)