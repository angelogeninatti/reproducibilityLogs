import multiprocessing
import os
import pickle
from collections import defaultdict
from typing import Dict, List

import mysql.connector
import numpy as np
import pandas as pd
import tqdm
from dotenv import load_dotenv
from scipy import stats
from scipy.stats import ttest_ind

from connection_data import *

load_dotenv()

from logs import Logs

# Add classic TOST test function
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


def analyze_questionnaire(timeline, excluded):
    questionnaire_data = defaultdict(list)
    for event in timeline.events:
        if 'answered_questions' in event.event_name and event.event_name.split("_")[-1] not in excluded:
            for question, answer in event.answers.items():
                likert_scale = {
                    "Strongly Disagree": 1,
                    "Disagree": 2,
                    "Neither Agree nor Disagree": 3,
                    "Agree": 4,
                    "Strongly Agree": 5
                }
                numeric_answer = likert_scale.get(answer, np.nan)
                questionnaire_data[question].append(numeric_answer)
    return questionnaire_data


def read_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def read_qrels(file_path: str, binarize_threshold: int = -1, remap_threshold: int = -1) -> Dict[str, Dict[str, int]]:
    qrels = defaultdict(lambda: defaultdict(int))
    with open(file_path, 'r') as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split()
            if binarize_threshold >= 0:
                relevance = 1 if int(relevance) >= binarize_threshold else 0
            elif remap_threshold >= 0:
                relevance = int(relevance) if int(relevance) <= remap_threshold else remap_threshold
            else:
                relevance = int(relevance)
            qrels[query_id][doc_id] = relevance
    return qrels


def average_precision(relevance: List[int]) -> float:
    relevant_count = 0
    total_precision = 0
    for i, rel in enumerate(relevance, 1):
        if rel > 0:
            relevant_count += 1
            total_precision += relevant_count / i
    return total_precision / relevant_count if relevant_count > 0 else 0


def ndcg_at_k(ground_truth, k):
    dcg = sum((2 ** rel - 1) / np.log2(idx + 2)
              for idx, rel in enumerate(ground_truth[:k]))

    # Calculate IDCG@k using sorted ground truth
    sorted_relevance = sorted(ground_truth, reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / np.log2(idx + 2)
               for idx, rel in enumerate(sorted_relevance))

    # Return NDCG@k, handling division by zero
    return dcg / idcg if idcg > 0 else 0


def calculate_metrics(timeline):
    qrels_binarized = read_qrels('2022.qrels.pass.withDupes.txt', binarize_threshold=2)

    metrics, exp_condition, num_included_queries, excluded = calculate_interaction_metrics(
        timeline, qrels_binarized)
    relevance_metrics = calculate_relevance_metrics(timeline, qrels_binarized)

    # Merge the metrics
    for click_type in metrics:
        metrics[click_type].update(relevance_metrics[click_type])

    return metrics, exp_condition, num_included_queries, excluded


def inner_metrics(log_type, relevances, metrics, k):
    relevances_at_k = relevances[:k]

    ndcg = ndcg_at_k(relevances, k)
    ap = average_precision(relevances_at_k)

    precision = sum(relevances_at_k) / k

    metrics[log_type]['ndcg@' + str(k)].append(ndcg)
    metrics[log_type]['ap@' + str(k)].append(ap)
    metrics[log_type]['precision@' + str(k)].append(precision)

    return metrics


def calculate_relevance_metrics(timeline, qrels_binarized):
    metrics = defaultdict(lambda: defaultdict(list))

    idx = -1
    for event in timeline.get_logs("search_results"):
        try:
            skipped = timeline.get_next_log_after_event('skipped_query', event)
            if skipped['query']['query_id'] == event['query']['query_id']:
                continue
        except FileNotFoundError:
            pass
        last_ground_truth_relevances = [qrels_binarized[str(event['query']['query_id'])].get(result['docno'], 0) for
                                        result in
                                        event['results']]

        # If last_ground_truth_relevance is all 0s or all 1s, skip the query
        if all(rel == 0 for rel in last_ground_truth_relevances):
            continue

        idx += 1
        try:
            next_search_or_end = timeline.get_next_log_after_event('search_results', event)
        except FileNotFoundError:
            next_search_or_end = timeline.get_next_log_after_event('end_experiment', event)
        expand_logs = timeline.get_events_between(event, next_search_or_end, 'expand_result')
        expanded_relevances = {}
        for expand_log in expand_logs:
            if expanded_relevances.get(expand_log['index']) is not None:
                continue
            expanded_relevances[expand_log['index']] = 1
        expanded_relevances = [expanded_relevances.get(i, 0) for i in range(len(last_ground_truth_relevances))]

        final_choose_log = timeline.get_next_log_after_event('confirmed', event)
        chosen_indices = final_choose_log['chosen_indices']  # Like [0,1,4]
        chosen_relevance = [1 if i in chosen_indices else 0 for i in range(len(last_ground_truth_relevances))]
        metrics = inner_metrics('expand', expanded_relevances, metrics, 5)
        metrics = inner_metrics('choose', chosen_relevance, metrics, 5)
    return metrics


def calculate_interaction_metrics(timeline, qrels):
    num_included_queries = 0
    metrics = defaultdict(lambda: defaultdict(list))
    exp_condition = None
    idx = -1
    excluded = []

    for event in timeline.events:
        if event.event_name == 'user_created':
            exp_condition = event.log_information.get('exp_condition')
        elif event.event_name == 'search_results':
            try:
                skipped = timeline.get_next_log_after_event('skipped_query', event)
                if skipped['query']['query_id'] == event['query']['query_id']:
                    continue
            except FileNotFoundError:
                pass
            # If last_ground_truth_relevance is all 0s or all 1s, skip the query
            last_ground_truth_relevances = [qrels[str(event['query']['query_id'])].get(result['docno'], 0) for result in
                                            event['results']]
            if all(rel == 0 for rel in last_ground_truth_relevances):
                excluded.append(event["exp_step"])
                continue
            idx += 1
            num_included_queries += 1

            start_time = event.timestamp
            search_results = event.log_information['results']
            choose_clicks = []
            expand_clicks = set()
            last_choose_click_time = None
            last_expand_click_time = None
            first_expand_click_time = None

            for next_event in timeline.events[timeline.events.index(event) + 1:]:
                if next_event.event_name == 'choose_result':
                    choose_click_index = next_event.log_information['index']
                    confirmed_event = timeline.get_next_log_after_event('confirmed', next_event)
                    if confirmed_event and 'chosen_results' in confirmed_event.log_information:
                        chosen_indices = [result['rank'] for result in
                                          confirmed_event.log_information['chosen_results']]
                        if choose_click_index in chosen_indices:
                            choose_clicks.append(choose_click_index)
                            last_choose_click_time = next_event.timestamp
                elif next_event.event_name == 'expand_result':
                    if first_expand_click_time is None:
                        first_expand_click_time = next_event.timestamp
                    expand_clicks.add(next_event.log_information['index'])
                    last_expand_click_time = next_event.timestamp
                elif next_event.event_name == 'confirmed':
                    end_time = next_event.timestamp
                    break

            # Find first choose_result event
            first_choose_time = timeline.get_next_log_after_event('choose_result', event).timestamp
            time_to_first_choose = (first_choose_time - start_time).total_seconds()
            metrics['choose']['time_to_first_choose'].append(time_to_first_choose)

            # Initialize time_to_first_choose for other click types with empty lists
            metrics['expand']['time_to_first_choose'].append(time_to_first_choose)

            expand_clicks = list(expand_clicks)
            for click_type, clicks in [('choose', choose_clicks), ('expand', expand_clicks)]:
                if clicks:
                    metrics[click_type]['mrr'].append(np.mean([1 / (rank + 1) for rank in clicks]))
                    metrics[click_type]['ctr@5'].append(len([c for c in clicks if c < 5]) / 5)

                    time_spent = (end_time - start_time).total_seconds()
                    metrics[click_type]['time_spent'].append(min(time_spent, 1800))  # Cap at 30 minutes

                    if click_type == 'choose':
                        time_to_first_click = (last_choose_click_time - start_time).total_seconds()
                        time_to_last_click = (last_choose_click_time - start_time).total_seconds()
                    else:
                        time_to_first_click = (first_expand_click_time - start_time).total_seconds()
                        time_to_last_click = (last_expand_click_time - start_time).total_seconds()

                    metrics[click_type]['time_to_first_click'].append(time_to_first_click)
                    metrics[click_type]['time_to_last_click'].append(time_to_last_click)
    return metrics, exp_condition, num_included_queries, excluded


def create_default_metrics():
    return defaultdict(list)


def create_default_click_types():
    return defaultdict(create_default_metrics)


def create_default_conditions():
    return defaultdict(create_default_click_types)


def process_single_user(args):
    user_id, timeline = args

    metrics, exp_condition, num_included_queries, excluded = calculate_metrics(timeline)
    questionnaire_data = analyze_questionnaire(timeline, excluded)

    condition_key = f"b={exp_condition['b']},k1={exp_condition['k1']},batch_size={exp_condition['batch_size']}"

    # Convert defaultdict to regular dict for serialization
    metrics_dict = {
        click_type: {
            metric: list(values) for metric, values in click_metrics.items()
        } for click_type, click_metrics in metrics.items()
    }

    questionnaire_dict = {
        question: list(answers) for question, answers in questionnaire_data.items()
    }
    return (
        condition_key, metrics_dict, questionnaire_dict, num_included_queries, user_id)


def merge_results(result, all_metrics, all_questionnaire_data):
    if result is None:
        return

    condition_key, metrics, questionnaire_data, num_queries, user_id = result

    # Merge metrics
    for click_type, click_metrics in metrics.items():
        for metric, values in click_metrics.items():
            all_metrics[condition_key][click_type][metric].extend(values)

    # Merge questionnaire data
    for question, answers in questionnaire_data.items():
        all_questionnaire_data[condition_key][question].extend(answers)


def format_pvalue_table(test_results, metrics_list, click_type):
    # Define the conditions in order
    conditions = [
        "(0.5,0.8)",
        "(0.25,0.6)",
        "(0.8,0.95)",
        "(0.75,0.85)",
        "(0.2,0.5)",
        "(0.2,0.85)",
        "(0.65,1.1)"
    ]

    # Create a DataFrame with conditions as rows and metrics as columns
    df = pd.DataFrame(index=conditions, columns=metrics_list)

    # Convert test results to proper format
    for result in test_results:
        if result['label'] != click_type:
            continue

        # Convert condition ID to (b,k1) format
        cond = CONDITIONS[result['comparison_condition']]
        cond_str = f"({cond['b']},{cond['k1']})"

        # Only include if condition is in our ordered list
        if cond_str in conditions:
            df.at[cond_str, result['metric']] = result['t_test_p_value']

    # Fill any remaining NaN values with empty string
    df = df.fillna('')

    return df


def save_pvalue_tables(all_metrics, results_dir):
    # Get list of metrics
    sample_metrics = all_metrics["b=0.5,k1=0.8,batch_size=4"]['choose'].keys()
    metrics_list = sorted(list(sample_metrics))

    # Run statistical tests
    choose_results, _ = statistical_tests(choose_metrics, "choose", True, 10, 10)
    expand_results, _ = statistical_tests(expand_metrics, "expand", True, 10, 10)

    # Create tables
    choose_table = format_pvalue_table(choose_results, metrics_list, "choose")
    expand_table = format_pvalue_table(expand_results, metrics_list, "expand")

    # Save tables
    choose_table.to_csv(f"{results_dir}/pvalue_table_choose.csv")
    expand_table.to_csv(f"{results_dir}/pvalue_table_expand.csv")

    return choose_table, expand_table

def process_logs(logs_input=None, n_processes=None, force_reload=False):
    if os.path.exists("processed_logs.pkl") and not force_reload:
        with open("processed_logs.pkl", "rb") as f:
            print(
                "INFO: loading processed log data from pickle. If you need to start over, please delete processed_logs.pkl (and df_test_log.bin).")
            return tuple(pickle.load(f))
    logs = logs_input if logs_input is not None else Logs("test_log")
    # Initialize the base dictionaries with proper default factories
    all_metrics = create_default_conditions()
    all_questionnaire_data = defaultdict(lambda: defaultdict(list))

    # Prepare arguments for parallel processing
    args_list = []

    for user_id, timeline in logs.items():
        if len(timeline.get_logs("end_experiment")) > 0:
            args_list.append((user_id, timeline))

    # Create process pool
    if len(args_list) > 10:
        with multiprocessing.Pool(processes=n_processes) as pool:
            # Process items in parallel with progress bar
            results = list(tqdm.tqdm(
                pool.imap_unordered(process_single_user, args_list),
                total=len(args_list),
                bar_format='{l_bar}{bar}|'
            ))
            # Merge results
            for result in results:
                if result and result is not None:
                    merge_results(result, all_metrics, all_questionnaire_data)
    else:
        # Process as single result (multiprocessing is handled by the progressive analysis)
        for args in tqdm.tqdm(args_list, bar_format='{l_bar}{bar}|'):
            result = process_single_user(args)
            merge_results(result, all_metrics, all_questionnaire_data)

    with open("processed_logs.pkl", "wb") as f:
        regular_questionnaire_data = dict(
            (k, dict(v)) for k, v in all_questionnaire_data.items()
        )
        res = [all_metrics, regular_questionnaire_data]
        pickle.dump(res, f)
    return all_metrics, all_questionnaire_data


def get_condition_id(condition_str):
    """Convert condition string to condition ID based on CONDITIONS dictionary"""
    # Parse the condition string to extract parameters
    params = {}
    for param in condition_str.split(','):
        key, value = param.split('=')
        params[key.strip()] = float(value)

    # Find matching condition in CONDITIONS dictionary
    for cond_id, cond_params in CONDITIONS.items():
        if (cond_params['batch_size'] == params['batch_size'] and
                abs(cond_params['b'] - params['b']) < 1e-6 and
                abs(cond_params['k1'] - params['k1']) < 1e-6):
            return cond_id

    raise ValueError(f"No matching condition found for {condition_str}")


def save_basic_metrics(all_metrics, all_questionnaire_data):
    # Prepare data for basic metrics
    rows = []
    for condition, metrics in all_metrics.items():
        condition_id = get_condition_id(condition)
        for click_type in ['choose', 'expand']:
            for metric, values in metrics[click_type].items():
                rows.append({
                    'condition': condition_id,
                    'click_type': click_type,
                    'metric': metric,
                    'mean': round(np.mean(values), 2),
                    'std': round(np.std(values), 2)
                })

    # Save basic metrics
    df_basic = pd.DataFrame(rows)
    df_basic.to_csv(f"{results_dir}/basic_metrics.csv", index=False)

    # Save questionnaire results
    quest_rows = []
    for condition, quest_data in all_questionnaire_data.items():
        condition_id = get_condition_id(condition)
        for question, answers in quest_data.items():
            numeric_answers = [a for a in answers if not np.isnan(a)]
            if numeric_answers:
                quest_rows.append({
                    'condition': condition_id,
                    'question': questions.get(question, question),
                    'mean': round(np.mean(numeric_answers), 3),
                    'std': round(np.std(numeric_answers), 3)
                })

    df_quest = pd.DataFrame(quest_rows)
    df_quest.to_csv(f"{results_dir}/questionnaire_basic.csv", index=False)


def statistical_tests(metrics, label, progressive=False, round_ttest=4, round_tost=7):
    results = []
    tost_results = []

    for metric in metrics.keys():
        df = pd.DataFrame({
            'metric_value': metrics[metric],
            'condition': [get_condition_id(c) for c in conditions]
        })

        unique_conditions = list(set(conditions))
        base_condition = "b=0.5,k1=0.8,batch_size=4"
        base_condition_id = get_condition_id(base_condition)

        for condition in unique_conditions:
            if condition == base_condition:
                continue

            condition_id = get_condition_id(condition)
            base_values = df[df['condition'] == base_condition_id]['metric_value']
            comp_values = df[df['condition'] == condition_id]['metric_value']
            # T-test
            t_stat, t_p_value = stats.ttest_ind(base_values, comp_values, equal_var=False)

            results.append({
                'metric': metric,
                'label': label,
                'comparison_condition': condition_id,
                't_test_p_value': round(t_p_value, round_ttest),
            })

            # TOST analysis
            pooled_std = np.std(np.concatenate([base_values, comp_values]))
            epsilon = 0.25 * pooled_std
            classic_p = tost_test(base_values, comp_values, epsilon)

            tost_results.append({
                'metric': metric,
                'label': label,
                'comparison_condition': condition_id,
                'classic_tost_p_value': round(classic_p, round_tost),
            })

    # Save results
    if not progressive:
        pd.DataFrame(results).to_csv(f"{results_dir}/statistical_tests_{label}.csv", index=False)
        pd.DataFrame(tost_results).to_csv(f"{results_dir}/tost_tests_{label}.csv", index=False)
    return results, tost_results


def statistical_tests_questionnaire(questionnaire_data, label, questions):
    results = []

    df_quest = pd.DataFrame({
        'question': [],
        'answer': [],
        'condition': []
    })

    for condition, questions_data in questionnaire_data.items():
        condition_id = get_condition_id(condition)
        for question_id, answers in questions_data.items():
            question_text = questions.get(question_id, question_id)
            df_quest = pd.concat([df_quest, pd.DataFrame({
                'question': [question_text] * len(answers),
                'answer': answers,
                'condition': [condition_id] * len(answers)
            })])

    df_quest = df_quest.dropna()
    unique_questions = df_quest['question'].unique()
    base_condition = "b=0.5,k1=0.8,batch_size=4"
    base_condition_id = get_condition_id(base_condition)

    for question_text in unique_questions:
        question_data = df_quest[df_quest['question'] == question_text]
        unique_conditions = question_data['condition'].unique()

        for condition_id in unique_conditions:
            if condition_id == base_condition_id:
                continue

            base_values = question_data[question_data['condition'] == base_condition_id]['answer']
            comp_values = question_data[question_data['condition'] == condition_id]['answer']

            if len(base_values) == 0 or len(comp_values) == 0:
                continue

            t_stat, t_p_value = stats.ttest_ind(base_values, comp_values, equal_var=False)

            pooled_std = np.std(np.concatenate([base_values, comp_values]))
            epsilon = 0.25 * pooled_std
            classic_p = tost_test(base_values, comp_values, epsilon)

            results.append({
                'question': question_text,
                'comparison_condition': condition_id,
                't_test_p_value': round(t_p_value, 2),
                'classic_tost_p_value': round(classic_p, 3),
            })

    # Save results
    pd.DataFrame(results).to_csv(f"{results_dir}/questionnaire_statistical_tests.csv", index=False)


def create_db_connection(db="reproducibility"):
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password=os.getenv("DB_PWD"),
        database=db
    )


CONDITIONS = {
    0: {'batch_size': 4, 'b': 0.5, 'k1': 0.8},
    1: {'batch_size': 4, 'b': 0.2, 'k1': 0.5},
    2: {'batch_size': 4, 'b': 0.25, 'k1': 0.6},
    3: {'batch_size': 4, 'b': 0.65, 'k1': 1.1},
    4: {'batch_size': 4, 'b': 0.75, 'k1': 0.85},
    5: {'batch_size': 4, 'b': 0.8, 'k1': 0.95},
    7: {'batch_size': 4, 'b': 0.2, 'k1': 0.85}
}
if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()

    # Main execution
    db_logs = Logs("test_log")
    combined_logs = db_logs

    all_metrics, all_questionnaire_data = process_logs(combined_logs)

    choose_metrics = {metric: [] for metric in all_metrics["b=0.5,k1=0.8,batch_size=4"]['choose'].keys()}
    expand_metrics = {metric: [] for metric in all_metrics["b=0.5,k1=0.8,batch_size=4"]['choose'].keys()}

    conditions = []
    questions = {
        0: "All the results were relevant to the related query",
        1: "At least one result was relevant to the selected query",
        2: "The results were repetitive",
        3: "The results allowed me to satisfy the requested query",
        4: "The results were insufficient to satisfy my query",
    }

    for condition, metrics in all_metrics.items():
        for metric in choose_metrics.keys():
            choose_metrics[metric].extend(metrics['choose'][metric])
            expand_metrics[metric].extend(metrics['expand'][metric])
        conditions.extend([condition] * len(metrics['choose']['ndcg@5']))

    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    save_basic_metrics(all_metrics, all_questionnaire_data)

    # Perform statistical tests
    statistical_tests(choose_metrics, "choose")
    statistical_tests(expand_metrics, "expand")
    statistical_tests_questionnaire(all_questionnaire_data, "questionnaire", questions)

    choose_table, expand_table = save_pvalue_tables(all_metrics, results_dir)