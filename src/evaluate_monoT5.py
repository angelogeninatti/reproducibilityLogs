# import os
# import collections
# from tqdm import tqdm
# import pandas as pd
# import seaborn as sns
# from matplotlib import pyplot as plt
# import pyterrier as pt
# from repro_eval.Evaluator import RpdEvaluator
# from repro_eval.util import trim


# qrels = 'qrels/2022.qrels.pass.withDupes.txt'
# dirs = ['results', 'figures']


# def parse_run(f_run):
#     run = collections.defaultdict(dict)
#     for line in f_run:
#         query_id, _, object_id, ranking, score, _ = line.strip().split()
#         run[query_id][object_id] = float(score)
#     return run


# def parse_qrels(f_qrels):
#     qrels = collections.defaultdict(dict)
#     for line in f_qrels:
#         query_id, _, object_id, relevance = line.strip().split()
#         qrels[query_id][object_id] = int(relevance)
#     return qrels


# def evaluate_ktu(reference_file_name, B, K1, _t, _total, bs):
#     rpd_eval = RpdEvaluator(qrel_orig_path=qrels)
#     with open(reference_file_name) as f_in:
#         run_b_orig = parse_run(f_in)

#     rpd_eval.run_b_orig = run_b_orig
#     rpd_eval.trim(_t)
#     rpd_eval.evaluate()

#     ktu = {}

#     with tqdm(total=_total, position=0, leave=True) as pbar:
#         for b in B:
#             for k1 in K1:
#                 run_name = '_'.join(['runs/monoT5/txt/monoT5', str(bs), 'bm25', str(b), str(k1)])
#                 file_name = '.'.join([run_name, 'txt'])
#                 with open(file_name) as f_in:
#                     _run = parse_run(f_in)
#                     trim(_run, _t)
#                     ktu_values = rpd_eval.ktau_union(run_b_rep=_run).get('baseline').values()
#                     ktu[(b,k1)] = sum(ktu_values) / len(ktu_values)
#                     pbar.update()

#     df_data_ktu = {}
#     for b in B:
#         _data_ktu = {}
#         for k1 in K1:
#             _data_ktu[k1] = ktu.get((b, k1))
#         df_data_ktu[b] = _data_ktu

#     CMAP = 'coolwarm'
#     df_ktu = pd.DataFrame.from_dict(df_data_ktu).transpose()
#     df_ktu.to_csv('results/monoT5_' + str(bs) + '_bm25_ktu' + '_' + str(_t) + '.csv')
#     plt.figure(figsize=(4,3))
#     sns.heatmap(df_ktu, cmap=CMAP, cbar_kws={'label': r"Kendall's $\tau$ Union"}, vmin=.0, vmax=1.)
#     plt.ylabel("b")
#     plt.xlabel("k1")
#     plt.savefig('figures/monoT5_' + str(bs) + '_bm25_ktu_' + str(_t) + '.pdf', format='pdf', bbox_inches='tight')
#     plt.close()


# def evaluate_rmse(reference_file_name, B, K1, _t, _total, bs):
#     rpd_eval = RpdEvaluator(qrel_orig_path=qrels)
#     with open(reference_file_name) as f_in:
#         run_b_orig = parse_run(f_in)

#     rpd_eval.run_b_orig = run_b_orig
#     rpd_eval.trim(_t)
#     rpd_eval.evaluate()

#     rmse = {}

#     with tqdm(total=_total, position=0, leave=True) as pbar:
#         for b in B:
#             for k1 in K1:
#                 run_name = '_'.join(['runs/monoT5/txt/monoT5', str(bs), 'bm25', str(b), str(k1)])
#                 file_name = '.'.join([run_name, 'txt'])
#                 with open(file_name) as f_in:
#                     _run = parse_run(f_in)
#                     _rpd_eval = RpdEvaluator(qrel_orig_path=qrels)
#                     _rpd_eval.run_b_orig = _run
#                     _rpd_eval.trim(_t)
#                     _rpd_eval.evaluate()
#                     rmse[(b, k1)] = rpd_eval.nrmse(run_b_score=_rpd_eval.run_b_orig_score).get('baseline').get('P_10')
#                     pbar.update()

#     df_data_rmse = {}
#     for b in B:
#         _data_rmse = {}
#         for k1 in K1:
#             _data_rmse[k1] = rmse.get((b, k1))
#         df_data_rmse[b] = _data_rmse

#     CMAP = 'coolwarm'
#     df_rmse = pd.DataFrame.from_dict(df_data_rmse).transpose()
#     df_rmse.to_csv('results/monoT5_' + str(bs) + '_bm25_rmse' + '_' + str(_t) + '.csv')
#     plt.figure(figsize=(4,3))
#     sns.heatmap(df_rmse, cmap=CMAP, cbar_kws={'label': 'Normalized RMSE'}, vmin=0.0, vmax=0.25)
#     plt.ylabel("b")
#     plt.xlabel("k1")
#     plt.savefig('figures/monoT5_' + str(bs) + '_bm25_rmse_' + str(_t) + '.pdf', format='pdf', bbox_inches='tight')
#     plt.close()


# def evaluate_arp(reference_file_name, B, K1, _t, _total, bs):
#     rpd_eval = RpdEvaluator(qrel_orig_path=qrels)
#     with open(reference_file_name) as f_in:
#         run_b_orig = parse_run(f_in)

#     rpd_eval.run_b_orig = run_b_orig
#     rpd_eval.trim(_t)
#     rpd_eval.evaluate()

#     p10 = {}
#     with tqdm(total=_total, position=0, leave=True) as pbar:
#         for b in B:
#             for k1 in K1:
#                 run_name = '_'.join(['runs/monoT5/txt/monoT5', str(bs), 'bm25', str(b), str(k1)])
#                 file_name = '.'.join([run_name, 'txt'])
#                 with open(file_name) as f_in:
#                     _run = parse_run(f_in)
#                     rpd_eval.run_b_orig = _run
#                     rpd_eval.trim(_t)
#                     rpd_eval.evaluate()
#                     p10_scores = [t.get('P_10') for t in rpd_eval.run_b_orig_score.values()]
#                     p10[(b, k1)] = sum(p10_scores) / len(p10_scores)
#                     pbar.update()

#     df_data_p10 = {}
#     for b in B:
#         _data_p10 = {}
#         for k1 in K1:
#             _data_p10[k1] = p10.get((b, k1))
#         df_data_p10[b] = _data_p10

#     CMAP = 'coolwarm'
#     df_p10 = pd.DataFrame.from_dict(df_data_p10).transpose()
#     df_p10.to_csv('results/monoT5_' + str(bs) + '_bm25_p_10.csv')
#     plt.figure(figsize=(4,3))
#     sns.heatmap(df_p10, cmap=CMAP, cbar_kws={'label': 'P@10'}, vmin=.0, vmax=1.)
#     plt.ylabel("b")
#     plt.xlabel("k1")
#     plt.savefig('figures/monoT5_' + str(bs) + '_bm25_p_10.pdf', format='pdf', bbox_inches='tight')
#     plt.close()


# def main():
#     for dir in dirs:
#         os.makedirs(dir, exist_ok=True)
#     B = [round(0.2 + 0.05 * i, 3) for i in range(0,13)]
#     K1 = [round(0.5 + 0.05 * i, 3) for i in range(0,13)]
#     _total = len(B)*len(K1)
#     _t = 10
#     bs = 4
#     reference = (0.5, 0.8)
#     reference_run_name = '_'.join(['runs/monoT5/txt/monoT5', str(bs), 'bm25', str(reference[0]), str(reference[1])])
#     reference_file_name = '.'.join([reference_run_name, 'txt'])
#     evaluate_ktu(reference_file_name, B, K1, _t, _total, bs)
#     evaluate_rmse(reference_file_name, B, K1, _t, _total, bs)
#     evaluate_arp(reference_file_name, B, K1, _t, _total, bs)


# if __name__ == '__main__':
#     main()


import os
import collections
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pyterrier as pt
from repro_eval.Evaluator import RpdEvaluator
from repro_eval.util import trim
import ir_datasets


selected_queries_only = True
exclude = ['2033470', '2025747', '2055795', '2046371', '2002533', '2032956', '2003322', '2038890', '2034676', '2005861', '2006211', '2006627', '2007055', '2007419', '2008871', '2032090', '2049687', '2017299']
selection = []
dataset = ir_datasets.load("msmarco-passage-v2/trec-dl-2022/judged")
for query in dataset.queries_iter():
    if query.query_id not in exclude:
        selection.append(query.query_id)

if selected_queries_only:
    qrels = 'qrels/2022.qrels.pass.withDupes.stripped.txt'
else:
    qrels = 'qrels/2022.qrels.pass.withDupes.txt'

dirs = ['results', 'figures']


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


def evaluate_ktu(reference_file_name, B, K1, _t, _total, bs):
    rpd_eval = RpdEvaluator(qrel_orig_path=qrels)
    with open(reference_file_name) as f_in:
        run_b_orig = parse_run(f_in)

    rpd_eval.run_b_orig = run_b_orig
    rpd_eval.trim(_t)
    rpd_eval.evaluate()

    ktu = {}

    with tqdm(total=_total, position=0, leave=True) as pbar:
        for b in B:
            for k1 in K1:
                run_name = '_'.join(['runs/monoT5/txt/monoT5', str(bs), 'bm25', str(b), str(k1)])
                file_name = '.'.join([run_name, 'txt'])
                with open(file_name) as f_in:
                    _run = parse_run(f_in)
                    trim(_run, _t)
                    if selected_queries_only:
                        qs = rpd_eval.ktau_union(run_b_rep=_run).get('baseline')
                        ktu_values = {key: qs[key] for key in selection if key in qs}.values()
                    else:
                        ktu_values = rpd_eval.ktau_union(run_b_rep=_run).get('baseline').values()
                    ktu[(b,k1)] = sum(ktu_values) / len(ktu_values)
                    pbar.update()

    df_data_ktu = {}
    for b in B:
        _data_ktu = {}
        for k1 in K1:
            _data_ktu[k1] = ktu.get((b, k1))
        df_data_ktu[b] = _data_ktu

    CMAP = 'coolwarm'
    df_ktu = pd.DataFrame.from_dict(df_data_ktu).transpose()
    df_ktu.to_csv('results/monoT5_' + str(bs) + '_bm25_ktu' + '_' + str(_t) + '.csv')
    plt.figure(figsize=(4,3))
    sns.heatmap(df_ktu, cmap=CMAP, cbar_kws={'label': r"Kendall's $\tau$ Union"}, vmin=.0, vmax=1.)
    plt.ylabel("b")
    plt.xlabel("k1")
    plt.savefig('figures/monoT5_' + str(bs) + '_bm25_ktu_' + str(_t) + '.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def evaluate_rmse(reference_file_name, B, K1, _t, _total, bs):
    rpd_eval = RpdEvaluator(qrel_orig_path=qrels)
    with open(reference_file_name) as f_in:
        run_b_orig = parse_run(f_in)

    rpd_eval.run_b_orig = run_b_orig
    rpd_eval.trim(_t)
    rpd_eval.evaluate()

    rmse = {}

    with tqdm(total=_total, position=0, leave=True) as pbar:
        for b in B:
            for k1 in K1:
                run_name = '_'.join(['runs/monoT5/txt/monoT5', str(bs), 'bm25', str(b), str(k1)])
                file_name = '.'.join([run_name, 'txt'])
                with open(file_name) as f_in:
                    _run = parse_run(f_in)
                    _rpd_eval = RpdEvaluator(qrel_orig_path=qrels)
                    _rpd_eval.run_b_orig = _run
                    _rpd_eval.trim(_t)
                    _rpd_eval.evaluate()
                    rmse[(b, k1)] = rpd_eval.nrmse(run_b_score=_rpd_eval.run_b_orig_score).get('baseline').get('P_5')
                    pbar.update()

    df_data_rmse = {}
    for b in B:
        _data_rmse = {}
        for k1 in K1:
            _data_rmse[k1] = rmse.get((b, k1))
        df_data_rmse[b] = _data_rmse

    CMAP = 'coolwarm'
    df_rmse = pd.DataFrame.from_dict(df_data_rmse).transpose()
    df_rmse.to_csv('results/monoT5_' + str(bs) + '_bm25_rmse' + '_' + str(_t) + '.csv')
    plt.figure(figsize=(4,3))
    sns.heatmap(df_rmse, cmap=CMAP, cbar_kws={'label': 'Normalized RMSE'}, vmin=0.0, vmax=0.25)
    plt.ylabel("b")
    plt.xlabel("k1")
    plt.savefig('figures/monoT5_' + str(bs) + '_bm25_rmse_' + str(_t) + '.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def evaluate_arp(reference_file_name, B, K1, _t, _total, bs):
    rpd_eval = RpdEvaluator(qrel_orig_path=qrels)
    with open(reference_file_name) as f_in:
        run_b_orig = parse_run(f_in)

    rpd_eval.run_b_orig = run_b_orig
    rpd_eval.trim(_t)
    rpd_eval.evaluate()

    p10 = {}
    with tqdm(total=_total, position=0, leave=True) as pbar:
        for b in B:
            for k1 in K1:
                run_name = '_'.join(['runs/monoT5/txt/monoT5', str(bs), 'bm25', str(b), str(k1)])
                file_name = '.'.join([run_name, 'txt'])
                with open(file_name) as f_in:
                    _run = parse_run(f_in)
                    rpd_eval.run_b_orig = _run
                    rpd_eval.trim(_t)
                    rpd_eval.evaluate()
                    p10_scores = [t.get('P_5') for t in rpd_eval.run_b_orig_score.values()]
                    p10[(b, k1)] = sum(p10_scores) / len(p10_scores)
                    pbar.update()

    df_data_p10 = {}
    for b in B:
        _data_p10 = {}
        for k1 in K1:
            _data_p10[k1] = p10.get((b, k1))
        df_data_p10[b] = _data_p10

    CMAP = 'coolwarm'
    df_p10 = pd.DataFrame.from_dict(df_data_p10).transpose()
    df_p10.to_csv('results/monoT5_' + str(bs) + '_bm25_p_5.csv')
    plt.figure(figsize=(4,3))
    sns.heatmap(df_p10, cmap=CMAP, cbar_kws={'label': 'P@5'}, vmin=.0, vmax=1.)
    plt.ylabel("b")
    plt.xlabel("k1")
    plt.savefig('figures/monoT5_' + str(bs) + '_bm25_p_5.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def main():
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
    B = [round(0.2 + 0.05 * i, 3) for i in range(0,13)]
    K1 = [round(0.5 + 0.05 * i, 3) for i in range(0,13)]
    _total = len(B)*len(K1)
    _t = 5
    bs = 4 # 4, 10, 50
    reference = (0.5, 0.8)
    reference_run_name = '_'.join(['runs/monoT5/txt/monoT5', str(bs), 'bm25', str(reference[0]), str(reference[1])])
    reference_file_name = '.'.join([reference_run_name, 'txt'])
    evaluate_ktu(reference_file_name, B, K1, _t, _total, bs)
    evaluate_rmse(reference_file_name, B, K1, _t, _total, bs)
    evaluate_arp(reference_file_name, B, K1, _t, _total, bs)

if __name__ == '__main__':
    main()
