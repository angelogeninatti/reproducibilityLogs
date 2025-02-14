import random
import numpy as np
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
        self.continuation = []
        self.page_length = page_length
        self.click_logs = pd.read_csv(click_logs_path)
        self.qrels = pd.read_csv(qrels_path, names=['qid', 'Q0', 'docid', 'rel'], sep=' ')

    def update_continuation(self):
        """
        Determine the continuation parameter based on the logged clicks.
        """
        last_clicks = [0] * self.page_length
        for last_click in self.click_logs.groupby(['user_id', 'query_id'])['clicked_document_index'].max():
            last_clicks[last_click] += 1

        total_clicks = [0] * self.page_length
        for i in range(0, self.page_length):
            total_clicks[i] = len(self.click_logs[self.click_logs['clicked_document_index'] == i])

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


def main():
    dcm = DCM(page_length=5, 
              click_logs_path='../click_logs.csv', 
              qrels_path='../qrels/2022.qrels.pass.withDupes.txt')
    dcm.update_continuation()
    dcm.update_relevance()

    qid = 2027130
    results = ["msmarco_passage_05_837259438",
               "msmarco_passage_00_702052494",
               "msmarco_passage_66_11162916",
               "msmarco_passage_43_353320329",
               "msmarco_passage_41_185345572"]

    clicks = dcm.simulate_clicks(qid, results)
    print(clicks)


if __name__ == '__main__':
    main()
