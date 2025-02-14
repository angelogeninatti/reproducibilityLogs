import pandas as pd 

from repro_eval.Evaluator import RpdEvaluator, RplEvaluator
from repro_eval.util import arp, arp_scores, print_base_adv, print_simple_line, trim

from scipy import stats
from scipy.stats import ttest_ind
import numpy as np


def tost_test(x, y, epsilon):
    t_stat, _ = ttest_ind(x, y)
    df = len(x) + len(y) - 2
    se = np.sqrt(np.var(x, ddof=1) / len(x) + np.var(y, ddof=1) / len(y))

    t_lower = (np.mean(x) - np.mean(y) + epsilon) / se
    t_upper = (np.mean(x) - np.mean(y) - epsilon) / se

    p_lower = 1 - stats.t.cdf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)

    p_value = max(p_lower, p_upper)

    return p_value


with open('results/used_queries.txt', 'r') as f_in:
    content = f_in.read()

queries = set(map(int, content.strip("{} \n").split(",")))
qrels = './data/qrels/2022.qrels.pass.withDupes.txt'
original = './runs/monoT5/plain/monoT5_4_bm25_0.5_0.8'
results = []

reproductions = {
    'orig': './runs/monoT5/plain/monoT5_4_bm25_0.5_0.8',
    'repone': './runs/monoT5/plain/monoT5_4_bm25_0.25_0.6',
    'reptwo': './runs/monoT5/plain/monoT5_4_bm25_0.8_0.95',
    'repthree': './runs/monoT5/plain/monoT5_4_bm25_0.75_0.85',
    'repfour': './runs/monoT5/plain/monoT5_4_bm25_0.2_0.5',
    'repfive': './runs/monoT5/plain/monoT5_4_bm25_0.2_0.85',
    'repsix': './runs/monoT5/plain/monoT5_4_bm25_0.65_1.1'
}

for name, path in reproductions.items():

    rpd_eval = RpdEvaluator(qrel_orig_path=qrels,
                            run_b_orig_path=original,
                            run_b_rep_path=path)

    rpd_eval.run_b_orig =  {str(key): rpd_eval.run_b_orig[str(key)] for key in queries}
    rpd_eval.run_b_rep =  {str(key): rpd_eval.run_b_rep[str(key)] for key in queries}
    rpd_eval.trim()
    rpd_eval.evaluate()

    recip_rank_scores_orig =[rpd_eval.run_b_orig_score.get(str(qid)).get('recip_rank') for qid in queries]
    P_5_scores_orig = [rpd_eval.run_b_orig_score.get(str(qid)).get('P_5') for qid in queries]
    ndcg_cut_5_scores_orig = [rpd_eval.run_b_orig_score.get(str(qid)).get('ndcg_cut_5') for qid in queries]
    map_cut_5_scores_orig = [rpd_eval.run_b_orig_score.get(str(qid)).get('map_cut_5') for qid in queries]

    recip_rank_scores =[rpd_eval.run_b_rep_score.get(str(qid)).get('recip_rank') for qid in queries]
    P_5_scores = [rpd_eval.run_b_rep_score.get(str(qid)).get('P_5') for qid in queries]
    ndcg_cut_5_scores = [rpd_eval.run_b_rep_score.get(str(qid)).get('ndcg_cut_5') for qid in queries]
    map_cut_5_scores = [rpd_eval.run_b_rep_score.get(str(qid)).get('map_cut_5') for qid in queries]
                           
    recip_rank = np.average(recip_rank_scores)
    P_5 = np.average(P_5_scores)
    ndcg_cut_5 = np.average(ndcg_cut_5_scores)
    map_cut_5 = np.average(map_cut_5_scores)

    recip_rank_std = np.std(recip_rank_scores)
    P_5_std = np.std(P_5_scores)
    ndcg_cut_5_std = np.std(ndcg_cut_5_scores)
    map_cut_5_std = np.std(map_cut_5_scores)

    rmse_recip_rank = rpd_eval.nrmse().get('baseline').get('recip_rank') 
    rmse_P_5 = rpd_eval.nrmse().get('baseline').get('P_5') 
    rmse_ndcg_cut_5 = rpd_eval.nrmse().get('baseline').get('ndcg_cut_5')
    rmse_map_cut_5 = rpd_eval.nrmse().get('baseline').get('map_cut_5')

    pval_recip_rank = rpd_eval.ttest().get('baseline').get('recip_rank') 
    pval_P_5 = rpd_eval.ttest().get('baseline').get('P_5') 
    pval_ndcg_cut_5 = rpd_eval.ttest().get('baseline').get('ndcg_cut_5') 
    pval_map_cut_5 = rpd_eval.ttest().get('baseline').get('map_cut_5') 


    pooled_std = np.std(np.concatenate([recip_rank_scores_orig, recip_rank_scores]))
    epsilon = 0.25 * pooled_std
    pval_tost_recip_rank = tost_test(recip_rank_scores_orig, recip_rank_scores, epsilon)
    
    pooled_std = np.std(np.concatenate([P_5_scores_orig, P_5_scores]))
    epsilon = 0.25 * pooled_std
    pval_tost_P_5 = tost_test(P_5_scores_orig, P_5_scores, epsilon)

    pooled_std = np.std(np.concatenate([ndcg_cut_5_scores_orig, ndcg_cut_5_scores]))
    epsilon = 0.25 * pooled_std
    pval_tost_ndcg_cut_5 = tost_test(ndcg_cut_5_scores_orig, ndcg_cut_5_scores, epsilon)

    pooled_std = np.std(np.concatenate([map_cut_5_scores_orig, map_cut_5_scores]))
    epsilon = 0.25 * pooled_std
    pval_tost_map_cut_5 = tost_test(map_cut_5_scores_orig, map_cut_5_scores, epsilon)

    results.append(
        {
            'ID': name,
            'MRR': recip_rank,
            'P@5': P_5, 
            'nDCG@5': ndcg_cut_5, 
            'AP@5': map_cut_5,
            'MRR (std)': recip_rank_std,
            'P@5 (std)': P_5_std, 
            'nDCG@5 (std)': ndcg_cut_5_std, 
            'AP@5 (std)': map_cut_5_std,
            '$RMSE_{MRR}$': rmse_recip_rank,
            '$RMSE_{P@5}$': rmse_P_5, 
            '$RMSE_{nDCG@5}$': rmse_ndcg_cut_5,
            '$RMSE_{AP@5}$': rmse_map_cut_5,
            '$p_{MRR}$': pval_recip_rank,
            '$p_{P@5}$': pval_P_5,
            '$p_{nDCG@5}$': pval_ndcg_cut_5,
            '$p_{AP@5}$': pval_map_cut_5,
            '$p_{MRR, TOST}$': pval_tost_recip_rank,
            '$p_{P@5, TOST}$': pval_tost_P_5,
            '$p_{nDCG@5, TOST}$': pval_tost_ndcg_cut_5,
            '$p_{AP@5, TOST}$': pval_tost_map_cut_5,
        }
    )

df = pd.DataFrame(data=results)
df.to_csv('./results/offline_reproduciblity.csv', index=0)
print(df.to_latex(float_format="%.4f", index=False))
