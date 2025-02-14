import json 

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
data

with open('./results/qid_ktu_store.json', 'w') as f_out:
    f_out.write(json.dumps(data))