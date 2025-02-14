from collections import defaultdict, Counter
from typing import Dict

from logs import Logs

def read_qrels(file_path: str) -> Dict[str, Dict[str, int]]:
    qrels = defaultdict(lambda: defaultdict(int))
    with open(file_path, 'r') as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split()
            qrels[query_id][doc_id] = int(relevance)
    return qrels


db_logs = Logs("test_log")
qrels = read_qrels("2022.qrels.pass.withDupes.txt")
relevant_chosen = 0
total_chosen = 0
non_relevant_discarded = 0
total_discarded = 0

for user_id, timeline in db_logs.items():
    confirm_logs = timeline.get_logs('confirmed')
    for confirmed_log in confirm_logs:
        chosen_logs = confirmed_log['chosen_results']
        for log in chosen_logs:
            query_id = log['query_id']
            doc_id = log['docno']
            relevance = qrels[str(query_id)].get(doc_id, 0)
            if relevance > 0:
                relevant_chosen += 1
            total_chosen += 1

print("Percentage of chosen items which were relevant:", relevant_chosen / total_chosen)