#!/usr/bin/env python3

import json 
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


bs = [4, 10, 50]
bs = [4]
B = [round(0.2 + 0.05 * i, 3) for i in range(0,13)]
K1 = [round(0.5 + 0.05 * i, 3) for i in range(0,13)]

systems = []
for _bs in bs:
    for b in B:
        for k1 in K1:
            systems.append('({}, {}, {})'.format(_bs, b, k1))

_4 = [] 
_4_5 = []
_5_6 = []
_6_7 = []
_7_8 = []
_8_9 = []
_9 = []

for system in systems:
    scores = [rankings.get(qid).get(system).get('ktu') for qid in rankings.keys()]
    avg_ktu = float(sum(scores) / len(scores))
    if avg_ktu < .4:
        _4.append(system)
    if avg_ktu > .4 and avg_ktu < .5:
        _4_5.append(system)
    if avg_ktu > .5 and avg_ktu < .6:
        _5_6.append(system)
    if avg_ktu > .6 and avg_ktu < .7:
        _6_7.append(system)
    if avg_ktu > .7 and avg_ktu < .8:
        _7_8.append(system)
    if avg_ktu > .8 and avg_ktu < .9:
        _8_9.append(system)
    if avg_ktu > .9:
        _9.append(system)

print('System with avg. KTU below 0.4')
print(_4)
print('System with avg. KTU between 0.4 and 0.5')
print(_4_5)
print('System with avg. KTU between 0.5 and 0.6')
print(_5_6)
print('System with avg. KTU between 0.6 and 0.7')
print(_6_7)
print('System with avg. KTU between 0.7 and 0.8')
print(_7_8)
print('System with avg. KTU between 0.8 and 0.9')
print(_8_9)
print('System with avg. KTU above 0.9')
print(_9)
