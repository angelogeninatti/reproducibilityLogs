import random
import json
import pandas as pd


class DCM:
    def __init__(self, page_length, click_logs_path, qrels_path):
        """
        Initializes the DCM model.
        
        :param page_length: Length of the result page, i.e., number of items.
        :param click_logs_path: Path to the CSV files with click logs.
        :param qrels_path: Path to the qrels file.
        """
        self.relevance = {}
        self.continuation = [1] * 5
        self.page_length = page_length
        self.click_logs = pd.read_csv(click_logs_path)
        self.qrels = pd.read_csv(qrels_path, names=['qid', 'Q0', 'docid', 'rel'], sep=' ')

    def update_continuation(self, query_id: str=None, system_config: tuple=None, query_ktu: float=None):
        """
        Determine the continuation parameter based on the logged clicks.
        """
        last_clicks = [0] * self.page_length

        click_logs = self.click_logs

        if query_id:
            click_logs = click_logs[click_logs['query_id'] == query_id]
        
        if system_config:
            click_logs = click_logs[(click_logs['condition_b'] == system_config[0]) & (click_logs['condition_k1'] == system_config[1])]
        
        if query_ktu:
            click_logs = click_logs[click_logs['query_ktu'] == query_ktu]

        for last_click in click_logs.groupby(['user_id', 'query_id'])['clicked_document_index'].max():
            last_clicks[last_click] += 1

        total_clicks = [0] * self.page_length
        for i in range(0, self.page_length):
            total_clicks[i] = len(click_logs[click_logs['clicked_document_index'] == i])

        self.continuation = [1 - (l_c/t_c) for l_c, t_c in zip(last_clicks,total_clicks)]

    def update_relevance(self):
        """
        Determine the relevance parameter based on the relevance judgments in the qrels file.
        """
        max_rel = self.qrels['rel'].max()
        # for row in self.qrels.iterrows():
        #     self.relevance[(row[1].qid, row[1].docid)] = row[1].rel / max_rel

        # faster alternative to iterrows()
        for qid, docid, rel in zip(self.qrels['qid'], self.qrels['docid'], self.qrels['rel']):
            self.relevance[(qid, docid)] = rel / max_rel

    def get_click_probs(self, qid, results):
        """
        Return the click probabilities for a given query identifier and results list with passage identifiers.

        :param qid: Query identifier as integer value.
        :param results: Python list with strings that correspond to the passage identifiers.
        """
        click_probs = []
        for i, docid in enumerate(results):
            rel = self.relevance.get((qid,docid))
            if rel == None:
                rel = 0.0

            if len(self.relevance) == 0:
                rel = 1.0

            attractiveness = 1.0
            for j in range(0, i):
                _rel = self.relevance.get((qid,results[j]))
                if _rel == None:
                    _rel = 0.0
                attractiveness *= 1 - _rel + self.continuation[j] * _rel
            
            click_probs.append(rel * attractiveness)

        return click_probs

    def simulate_clicks(self, qid, results):
        """
        Simulate clicks for a given query identifier and results list with passage identifiers.

        :param qid: Query identifier as integer value.
        :param results: Python list with strings that correspond to the passage identifiers.
        """
        click_probs = []
        for i, docid in enumerate(results):
            rel = self.relevance.get((qid,docid))
            if rel == None:
                rel = 0.0

            attractiveness = 1.0
            for j in range(0, i):
                _rel = self.relevance.get((qid,results[j]))
                if _rel == None:
                    _rel = 0.0
                attractiveness *= 1 - _rel + self.continuation[j] * _rel
            
            click_probs.append(rel * attractiveness)

        clicks = []
        for i, _ in enumerate(results):
            if click_probs[i] > random.uniform(0.0, 1.0): 
                clicks.append(1)
            else:
                clicks.append(0)

        return clicks
    
    def get_unique_qids(self):
        return self.click_logs['query_id'].unique()

with open('./results/rankings.json') as f_in:
    rankings = json.loads(f_in.read())

systems = [(0.5, 0.8), (0.25, 0.6), (0.8, 0.95), (0.75, 0.85), (0.2, 0.5), (0.2,0.85), (0.65, 1.1)]
systems_ktu = [1.0, 0.4, 0.3967, 0.3897, 0.3793, 0.35, 0.3]

dcm = DCM(page_length=5, 
            click_logs_path='./sampled/confirm_choose_logs_sampled_1.csv', 
            qrels_path='./qrels/2022.qrels.pass.withDupes.txt')
dcm.update_continuation()
dcm.update_relevance()

for i in range(1,6):
    dcm = DCM(page_length=5, 
                click_logs_path='./sampled/confirm_choose_logs_sampled_{}.csv'.format(str(i)), 
                qrels_path='./qrels/2022.qrels.pass.withDupes.txt')
    dcm.update_continuation()
    dcm.update_relevance()
    with open('simulations/confirm_choose_logs_sampled_{}.csv'.format(str(i)), 'w') as f_out:
        f_out.write('user_id,query_id,query_results,clicked_document_id,clicked_document_index,condition_b,condition_k1,system_ktu,query_ktu\n')
        for _ in range(107): # approx. 300 users with 4 queries on avg. 300*4=1200 sessions. 5 times the amount are 6000 simulated sessions. 6000 sessions / 56 queries ~= 107
            for system, system_ktu in zip(systems, systems_ktu):
                params = '(4, {}, {})'.format(system[0], system[1])
                clicks = {}
                for qid in dcm.get_unique_qids():
                    results = [r.get('docno') for r in rankings.get(str(qid)).get(params).get('ranking')][:5]
                    clicked_indices = dcm.simulate_clicks(int(qid), results)
                    clicked_results = [results[i] for i,c in enumerate(clicked_indices) if c]
                    for c_i, c_r in zip(clicked_indices, clicked_results):
                        user_id = 'simulated'
                        query_id = str(qid)
                        query_results = '"{}"'.format(','.join(results))
                        clicked_document_id = c_r
                        clicked_document_index = str(results.index(c_r))
                        condition_b = str(system[0])
                        condition_k1 = str(system[1])
                        query_ktu = str(rankings.get(str(qid)).get(params).get('ktu'))
                        line = ','.join([user_id,query_id,query_results,clicked_document_id,clicked_document_index,condition_b,condition_k1,str(system_ktu),query_ktu])
                        f_out.write(''.join([line, '\n']))
    
    dcm = DCM(page_length=5, 
                click_logs_path='./sampled/expand_result_logs_sampled_{}.csv'.format(str(i)), 
                qrels_path='./qrels/2022.qrels.pass.withDupes.txt')
    dcm.update_continuation()
    dcm.update_relevance()
    with open('simulations/expand_result_logs_sampled_{}.csv'.format(str(i)), 'w') as f_out:
        f_out.write('user_id,query_id,query_results,clicked_document_id,clicked_document_index,condition_b,condition_k1,system_ktu,query_ktu\n')
        for _ in range(107): # approx. 300 users with 4 queries on avg. ~1200 sessions. 5 times the amount are 6000 sessions. 6000 sessions / 56 queries ~= 107
            for system, system_ktu in zip(systems, systems_ktu):
                params = '(4, {}, {})'.format(system[0], system[1])
                clicks = {}
                for qid in dcm.get_unique_qids():
                    results = [r.get('docno') for r in rankings.get(str(qid)).get(params).get('ranking')][:5]
                    clicked_indices = dcm.simulate_clicks(int(qid), results)
                    clicked_results = [results[i] for i,c in enumerate(clicked_indices) if c]
                    for c_i, c_r in zip(clicked_indices, clicked_results):
                        user_id = 'simulated'
                        query_id = str(qid)
                        query_results = '"{}"'.format(','.join(results))
                        clicked_document_id = c_r
                        clicked_document_index = str(results.index(c_r))
                        condition_b = str(system[0])
                        condition_k1 = str(system[1])
                        query_ktu = str(rankings.get(str(qid)).get(params).get('ktu'))
                        line = ','.join([user_id,query_id,query_results,clicked_document_id,clicked_document_index,condition_b,condition_k1,str(system_ktu),query_ktu])
                        f_out.write(''.join([line, '\n']))