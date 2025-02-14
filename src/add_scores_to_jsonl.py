import json
import collections
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pyterrier as pt
from repro_eval.Evaluator import RpdEvaluator
from repro_eval.util import trim

qrels = './qrels/2022.qrels.pass.withDupes.txt'

def parse_run(f_run):
    run = collections.defaultdict(dict)
    for line in f_run:
        query_id, _, object_id, ranking, score, _ = line.strip().split()
        run[query_id][object_id] = float(score)

    return run

def parse_qrels(f_qrels):
    qrels = collections.defaultdict(dict)
    for line in f_qrels:
        query_id, _, object_id, relevance = line.strip().split()
        qrels[query_id][object_id] = int(relevance)

    return qrels

B = [round(0.2 + 0.05 * i, 3) for i in range(0,13)]
K1 = [round(0.5 + 0.05 * i, 3) for i in range(0,13)]
_total = len(B)*len(K1)

_t = 5
reference = (0.5, 0.8)

reference_run_name = '_'.join(['./runs/bm25/txt/bm25', str(reference[0]), str(reference[1])])
reference_file_name = '.'.join([reference_run_name, 'txt'])
rpd_eval = RpdEvaluator(qrel_orig_path=qrels)
with open(reference_file_name) as f_in:
  run_b_orig = parse_run(f_in)
rpd_eval.run_b_orig = run_b_orig
rpd_eval.trim(_t)
rpd_eval.evaluate()

p5_scores_ref = {topic: scores.get('P_5') for topic, scores in rpd_eval.run_b_orig_score.items()}
p10_scores_ref = {topic: scores.get('P_10') for topic, scores in rpd_eval.run_b_orig_score.items()}
ndcg5_scores_ref = {topic: scores.get('ndcg_cut_5') for topic, scores in rpd_eval.run_b_orig_score.items()}
ndcg10_scores_ref = {topic: scores.get('ndcg_cut_10') for topic, scores in rpd_eval.run_b_orig_score.items()}

ktu = {}
P_5 = {}
arp_p10 = {}
delta_p10 = {}
ndcg_cut_5 = {}
arp_ndcg10 = {}
delta_ndcg10 = {}

print('Evaluate BM25 runs.')
with tqdm(total=_total, position=0, leave=True) as pbar:
  for b in B:
    for k1 in K1:
      run_name = '_'.join(['./runs/bm25/txt/bm25', str(b), str(k1)])
      file_name = '.'.join([run_name, 'txt'])
      with open(file_name) as f_in:
        _run = parse_run(f_in)
        trim(_run, _t)
        ktu[(b, k1)] = rpd_eval.ktau_union(run_b_rep=_run).get('baseline')
        P_5[(b, k1)] = p5_scores = {topic: scores.get('P_5') for topic, scores in rpd_eval.evaluate(run=_run).items()}
        arp_p10[(b, k1)] = p10_scores = {topic: scores.get('P_10') for topic, scores in rpd_eval.evaluate(run=_run).items()}
        ndcg_cut_5[(b, k1)] = ndcg5_scores = {topic: scores.get('ndcg_cut_5') for topic, scores in rpd_eval.evaluate(run=_run).items()}
        arp_ndcg10[(b, k1)] = ndcg10_scores = {topic: scores.get('ndcg_cut_10') for topic, scores in rpd_eval.evaluate(run=_run).items()}
        pbar.update()

for b_k1, scores in arp_ndcg10.items():
    deltas = {}
    for topic, score in scores.items():
        deltas[topic] = score - ndcg10_scores_ref.get(topic)
    delta_ndcg10[b_k1] = deltas

for b_k1, scores in arp_p10.items():
    deltas = {}
    for topic, score in scores.items():
        deltas[topic] = score - p10_scores_ref.get(topic)
    delta_p10[b_k1] = deltas

print('Annotate BM25 runs with scores.')
with tqdm(total=_total, position=0, leave=True) as pbar:
  for b in B:
    for k1 in K1:
      run_name = '_'.join(['./runs/bm25/jsonl/bm25', str(b), str(k1)])
      file_name = '.'.join([run_name, 'jsonl'])
      run_out = '_'.join(['./runs/bm25/jsonl/update/bm25', str(b), str(k1)])
      output_name = '.'.join([run_out, 'jsonl'])
      with open(file_name, 'r') as input_file, open(output_name, 'w') as output_file:
          for line in input_file:
            data = json.loads(line)
            data['ktu'] = ktu.get((b, k1)).get(data['qid'])
            data['P_5'] = P_5.get((b, k1)).get(data['qid'])
            data['ndcg_cut_5'] = ndcg_cut_5.get((b, k1)).get(data['qid'])
            # data['arp_p10'] = arp_p10.get((b, k1)).get(data['qid'])
            # data['arp_ndcg10'] = arp_ndcg10.get((b, k1)).get(data['qid'])
            # data['delta_p10'] = delta_p10.get((b, k1)).get(data['qid'])
            # data['delta_ndcg10'] = delta_ndcg10.get((b, k1)).get(data['qid'])
            json.dump(data, output_file)
            output_file.write('\n') 
      pbar.update() 

################################
# BM25 + monoT5 (batchsize 4) #
################################

reference_run_name = '_'.join(['./runs/monoT5/txt/monoT5_4_bm25', str(reference[0]), str(reference[1])])
reference_file_name = '.'.join([reference_run_name, 'txt'])
rpd_eval = RpdEvaluator(qrel_orig_path=qrels)
with open(reference_file_name) as f_in:
  run_b_orig = parse_run(f_in)
rpd_eval.run_b_orig = run_b_orig
rpd_eval.trim(_t)
rpd_eval.evaluate()

ktu = {}
P_5 = {}
arp_p10 = {}
delta_p10 = {}
ndcg_cut_5 = {}
arp_ndcg10 = {}
delta_ndcg10 = {}

print('Evaluate monoT5 runs with batch size of 4.')
with tqdm(total=_total, position=0, leave=True) as pbar:
  for b in B:
    for k1 in K1:
      run_name = '_'.join(['./runs/monoT5/txt/monoT5_4_bm25', str(b), str(k1)])
      file_name = '.'.join([run_name, 'txt'])
      with open(file_name) as f_in:
        _run = parse_run(f_in)
        trim(_run, _t)
        ktu[(b, k1)] = rpd_eval.ktau_union(run_b_rep=_run).get('baseline')
        P_5[(b, k1)] = p5_scores = {topic: scores.get('P_5') for topic, scores in rpd_eval.evaluate(run=_run).items()}
        arp_p10[(b, k1)] = p10_scores = {topic: scores.get('P_10') for topic, scores in rpd_eval.evaluate(run=_run).items()}
        ndcg_cut_5[(b, k1)] = ndcg5_scores = {topic: scores.get('ndcg_cut_5') for topic, scores in rpd_eval.evaluate(run=_run).items()}
        arp_ndcg10[(b, k1)] = ndcg10_scores = {topic: scores.get('ndcg_cut_10') for topic, scores in rpd_eval.evaluate(run=_run).items()}
        pbar.update()

for b_k1, scores in arp_ndcg10.items():
    deltas = {}
    for topic, score in scores.items():
        deltas[topic] = score - ndcg10_scores_ref.get(topic)
    delta_ndcg10[b_k1] = deltas

for b_k1, scores in arp_p10.items():
    deltas = {}
    for topic, score in scores.items():
        deltas[topic] = score - p10_scores_ref.get(topic)
    delta_p10[b_k1] = deltas

print('Annotate  monoT5 runs (batch size of 4) with scores.')
with tqdm(total=_total, position=0, leave=True) as pbar:
  for b in B:
    for k1 in K1:
      run_name = '_'.join(['./runs/monoT5/jsonl/monoT5_4_bm25', str(b), str(k1)])
      file_name = '.'.join([run_name, 'jsonl'])
      run_out = '_'.join(['./runs/monoT5/jsonl/update/monoT5_4_bm25', str(b), str(k1)])
      output_name = '.'.join([run_out, 'jsonl'])
      with open(file_name, 'r') as input_file, open(output_name, 'w') as output_file:
          for line in input_file:
            data = json.loads(line)
            data['ktu'] = ktu.get((b, k1)).get(data['qid'])
            data['P_5'] = P_5.get((b, k1)).get(data['qid'])
            data['ndcg_cut_5'] = ndcg_cut_5.get((b, k1)).get(data['qid'])
            # data['arp_p10'] = arp_p10.get((b, k1)).get(data['qid'])
            # data['arp_ndcg10'] = arp_ndcg10.get((b, k1)).get(data['qid'])
            # data['delta_p10'] = delta_p10.get((b, k1)).get(data['qid'])
            # data['delta_ndcg10'] = delta_ndcg10.get((b, k1)).get(data['qid'])
            json.dump(data, output_file)
            output_file.write('\n') 
      pbar.update() 

################################
# BM25 + monoT5 (batchsize 10) #
################################

reference_run_name = '_'.join(['./runs/monoT5/txt/monoT5_10_bm25', str(reference[0]), str(reference[1])])
reference_file_name = '.'.join([reference_run_name, 'txt'])
rpd_eval = RpdEvaluator(qrel_orig_path=qrels)
with open(reference_file_name) as f_in:
  run_b_orig = parse_run(f_in)
rpd_eval.run_b_orig = run_b_orig
rpd_eval.trim(_t)
rpd_eval.evaluate()

ktu = {}
P_5 = {}
arp_p10 = {}
delta_p10 = {}
ndcg_cut_5 = {}
arp_ndcg10 = {}
delta_ndcg10 = {}

print('Evaluate monoT5 runs with batch size of 10.')
with tqdm(total=_total, position=0, leave=True) as pbar:
  for b in B:
    for k1 in K1:
      run_name = '_'.join(['./runs/monoT5/txt/monoT5_10_bm25', str(b), str(k1)])
      file_name = '.'.join([run_name, 'txt'])
      with open(file_name) as f_in:
        _run = parse_run(f_in)
        trim(_run, _t)
        ktu[(b, k1)] = rpd_eval.ktau_union(run_b_rep=_run).get('baseline')
        P_5[(b, k1)] = p5_scores = {topic: scores.get('P_5') for topic, scores in rpd_eval.evaluate(run=_run).items()}
        arp_p10[(b, k1)] = p10_scores = {topic: scores.get('P_10') for topic, scores in rpd_eval.evaluate(run=_run).items()}
        ndcg_cut_5[(b, k1)] = ndcg5_scores = {topic: scores.get('ndcg_cut_5') for topic, scores in rpd_eval.evaluate(run=_run).items()}
        arp_ndcg10[(b, k1)] = ndcg10_scores = {topic: scores.get('ndcg_cut_10') for topic, scores in rpd_eval.evaluate(run=_run).items()}
        pbar.update()

for b_k1, scores in arp_ndcg10.items():
    deltas = {}
    for topic, score in scores.items():
        deltas[topic] = score - ndcg10_scores_ref.get(topic)
    delta_ndcg10[b_k1] = deltas

for b_k1, scores in arp_p10.items():
    deltas = {}
    for topic, score in scores.items():
        deltas[topic] = score - p10_scores_ref.get(topic)
    delta_p10[b_k1] = deltas

print('Annotate  monoT5 runs (batch size of 10) with scores.')
with tqdm(total=_total, position=0, leave=True) as pbar:
  for b in B:
    for k1 in K1:
      run_name = '_'.join(['./runs/monoT5/jsonl/monoT5_10_bm25', str(b), str(k1)])
      file_name = '.'.join([run_name, 'jsonl'])
      run_out = '_'.join(['./runs/monoT5/jsonl/update/monoT5_10_bm25', str(b), str(k1)])
      output_name = '.'.join([run_out, 'jsonl'])
      with open(file_name, 'r') as input_file, open(output_name, 'w') as output_file:
          for line in input_file:
            data = json.loads(line)
            data['ktu'] = ktu.get((b, k1)).get(data['qid'])
            data['P_5'] = P_5.get((b, k1)).get(data['qid'])
            data['ndcg_cut_5'] = ndcg_cut_5.get((b, k1)).get(data['qid'])
            # data['arp_p10'] = arp_p10.get((b, k1)).get(data['qid'])
            # data['arp_ndcg10'] = arp_ndcg10.get((b, k1)).get(data['qid'])
            # data['delta_p10'] = delta_p10.get((b, k1)).get(data['qid'])
            # data['delta_ndcg10'] = delta_ndcg10.get((b, k1)).get(data['qid'])
            json.dump(data, output_file)
            output_file.write('\n') 
      pbar.update() 

  
################################
# BM25 + monoT5 (batchsize 50) #
################################

reference_run_name = '_'.join(['./runs/monoT5/txt/monoT5_50_bm25', str(reference[0]), str(reference[1])])
reference_file_name = '.'.join([reference_run_name, 'txt'])
rpd_eval = RpdEvaluator(qrel_orig_path=qrels)
with open(reference_file_name) as f_in:
  run_b_orig = parse_run(f_in)
rpd_eval.run_b_orig = run_b_orig
rpd_eval.trim(_t)
rpd_eval.evaluate()

ktu = {}
P_5 = {}
arp_p10 = {}
delta_p10 = {}
ndcg_cut_5 = {}
arp_ndcg10 = {}
delta_ndcg10 = {}

print('Evaluate monoT5 runs with batch size of 50.')
with tqdm(total=_total, position=0, leave=True) as pbar:
  for b in B:
    for k1 in K1:
      run_name = '_'.join(['./runs/monoT5/txt/monoT5_50_bm25', str(b), str(k1)])
      file_name = '.'.join([run_name, 'txt'])
      with open(file_name) as f_in:
        _run = parse_run(f_in)
        trim(_run, _t)
        ktu[(b, k1)] = rpd_eval.ktau_union(run_b_rep=_run).get('baseline')
        P_5[(b, k1)] = p5_scores = {topic: scores.get('P_5') for topic, scores in rpd_eval.evaluate(run=_run).items()}
        arp_p10[(b, k1)] = p10_scores = {topic: scores.get('P_10') for topic, scores in rpd_eval.evaluate(run=_run).items()}
        ndcg_cut_5[(b, k1)] = ndcg5_scores = {topic: scores.get('ndcg_cut_5') for topic, scores in rpd_eval.evaluate(run=_run).items()}
        arp_ndcg10[(b, k1)] = ndcg10_scores = {topic: scores.get('ndcg_cut_10') for topic, scores in rpd_eval.evaluate(run=_run).items()}
        pbar.update()

for b_k1, scores in arp_ndcg10.items():
    deltas = {}
    for topic, score in scores.items():
        deltas[topic] = score - ndcg10_scores_ref.get(topic)
    delta_ndcg10[b_k1] = deltas

for b_k1, scores in arp_p10.items():
    deltas = {}
    for topic, score in scores.items():
        deltas[topic] = score - p10_scores_ref.get(topic)
    delta_p10[b_k1] = deltas

print('Annotate  monoT5 runs (batch size of 50) with scores.')
with tqdm(total=_total, position=0, leave=True) as pbar:
  for b in B:
    for k1 in K1:
      run_name = '_'.join(['./runs/monoT5/jsonl/monoT5_50_bm25', str(b), str(k1)])
      file_name = '.'.join([run_name, 'jsonl'])
      run_out = '_'.join(['./runs/monoT5/jsonl/update/monoT5_50_bm25', str(b), str(k1)])
      output_name = '.'.join([run_out, 'jsonl'])
      with open(file_name, 'r') as input_file, open(output_name, 'w') as output_file:
          for line in input_file:
            data = json.loads(line)
            data['ktu'] = ktu.get((b, k1)).get(data['qid'])
            data['P_5'] = P_5.get((b, k1)).get(data['qid'])
            data['ndcg_cut_5'] = ndcg_cut_5.get((b, k1)).get(data['qid'])
            # data['arp_p10'] = arp_p10.get((b, k1)).get(data['qid'])
            # data['arp_ndcg10'] = arp_ndcg10.get((b, k1)).get(data['qid'])
            # data['delta_p10'] = delta_p10.get((b, k1)).get(data['qid'])
            # data['delta_ndcg10'] = delta_ndcg10.get((b, k1)).get(data['qid'])
            json.dump(data, output_file)
            output_file.write('\n') 
      pbar.update() 