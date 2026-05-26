import random
import numpy as np
import pandas as pd
from collections import defaultdict


class DCTR:
    """
    DCTR (Document Click-Through Rate) Click Model.

    This model estimates the relevance of documents based on their empirical
    click-through rates (CTR) aggregated over user interactions for each
    (query_id, document_id) pair. It can also simulate clicks for a ranked
    document list given a query using the estimated CTRs.

    Attributes:
    -----------
    click_logs : pd.DataFrame
        A DataFrame containing the raw click logs
    dctr : dict
        A dictionary mapping (query_id, document_id) to its estimated CTR.
    """

    def __init__(self, click_logs_path):
        """
        Initialize the DCTR model with a path to the click logs CSV.

        Parameters:
        -----------
        click_logs_path : str
            Path to the CSV file containing click logs.
        """
        self.click_logs = pd.read_csv(click_logs_path)

    def update(self):
        """
        Estimate the DCTR values based on the current click logs.

        This method:
        - Groups logs by (user_id, query_id),
        - Parses the result list per query instance,
        - Tallies impressions and clicks per (query_id, document_id),
        - Computes click-through rate (CTR),
        - Stores a dictionary for quick access during simulation.
        """
        impression_counts = defaultdict(int)
        click_counts = defaultdict(int)

        # Group click logs by unique (user_id, query_id) pairs
        grouped = self.click_logs.groupby(['user_id', 'query_id'])

        for (user_id, query_id), group in grouped:
            # Assume all rows share the same result list for a given query
            result_list = group['query_results'].iloc[0].split(',')

            # Collect clicked documents for this interaction
            clicked_docs = set(group['clicked_document_id'].tolist())

            # Tally impressions and clicks
            for doc_id in result_list:
                key = (query_id, doc_id)
                impression_counts[key] += 1
                if doc_id in clicked_docs:
                    click_counts[key] += 1

        # Compute CTR as clicks / impressions
        dctr = {
            key: click_counts[key] / impression_counts[key]
            for key in impression_counts
        }

        # Store as a DataFrame (optional: for analysis)
        dctr_df = pd.DataFrame([
            {
                'query_id': query_id,
                'document_id': doc_id,
                'DCTR': dctr[(query_id, doc_id)],
                'clicks': click_counts[(query_id, doc_id)],
                'impressions': impression_counts[(query_id, doc_id)]
            }
            for (query_id, doc_id) in dctr
        ])

        # Store DCTR as dictionary for efficient lookup
        self.dctr = {
            (row['query_id'], row['document_id']): row['DCTR']
            for _, row in dctr_df.iterrows()
        }

    def simulate_clicks(self, qid, results):
        """
        Simulate clicks on a ranked result list using DCTR values.

        Parameters:
        -----------
        qid : str
            The query ID.
        results : list of str
            List of document IDs ranked by a retrieval system.

        Returns:
        --------
        clicks : list of int
            Simulated click list (1 for click, 0 for no click) for each result.
        """
        click_probs = []
        for r in results:
            dctr = self.dctr.get((qid, r))
            if dctr:
                click_probs.append(dctr)
            else:
                click_probs.append(0.0)

        clicks = []
        for i, _ in enumerate(results):
            if click_probs[i] > random.uniform(0.0, 1.0): 
                clicks.append(1)
            else:
                clicks.append(0)

        return clicks


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


class DBN:
    """
    DBN (Dynamic Bayesian Network) Click Model.

    This probabilistic click model estimates both document attractiveness
    and user satisfaction, accounting for the sequential nature of user
    examination behavior.

    Attributes:
    -----------
    click_logs : pd.DataFrame
        The raw click logs.

    attractiveness : dict
        Maps (query_id, document_id) pairs to their estimated attractiveness probabilities.

    satisfaction : dict
        Maps (query_id, document_id) pairs to their estimated satisfaction probabilities.
    """

    def __init__(self, click_logs_path):
        """
        Initialize the DBN model with a CSV file path.

        Parameters:
        -----------
        click_logs_path : str
            Path to the CSV file containing click log data.
        """
        self.click_logs = pd.read_csv(click_logs_path)
        self.attractiveness = None 
        self.satisfaction = None

    def _em(self, sessions, A, S, max_iter=10):
        """
        Expectation-Maximization algorithm for DBN parameter estimation.

        Parameters:
        -----------
        sessions : list of list of tuples
            Each session is a list of (query_id, doc_id, clicked) triples.

        A : dict
            Initial attractiveness values.

        S : dict
            Initial satisfaction values.

        max_iter : int
            Number of EM iterations.

        Returns:
        --------
        A, S : tuple of dicts
            Estimated attractiveness and satisfaction dictionaries.
        """
        for iteration in range(max_iter):
            # Accumulators for numerators and denominators in EM updates
            A_num = defaultdict(float)
            A_den = defaultdict(float)
            S_num = defaultdict(float)
            S_den = defaultdict(float)

            for session in sessions:
                examination = 1.0  # User starts by examining the first result
                for (query_id, doc_id, clicked) in session:
                    key = (query_id, doc_id)
                    a = A[key]
                    s = S[key]

                    # E-step: compute expected updates for this document
                    if clicked:
                        A_num[key] += examination
                        A_den[key] += examination
                        S_num[key] += examination
                        S_den[key] += examination
                        examination *= (1 - s)  # Continue if not satisfied
                    else:
                        A_den[key] += examination
                        examination *= (1 - a)  # No click: lower examination of next

            # M-step: update A and S parameters
            for key in A_den:
                if A_den[key] > 0:
                    A[key] = A_num[key] / A_den[key]
            for key in S_den:
                if S_den[key] > 0:
                    S[key] = S_num[key] / S_den[key]

        return A, S

    def update(self):
        """
        Run the EM algorithm on the click logs to estimate attractiveness and satisfaction.

        This method:
        - Groups data by (user_id, query_id),
        - Constructs sessions of documents and clicks,
        - Initializes parameters,
        - Trains the model using the EM procedure.
        """
        grouped = self.click_logs.groupby(['user_id', 'query_id'])

        sessions = []
        for (user_id, query_id), group in grouped:
            result_list = group['query_results'].iloc[0].split(',')
            clicked_docs = set(group['clicked_document_id'].tolist())

            # Construct a session: each document labeled by whether it was clicked
            session = [(query_id, doc_id, doc_id in clicked_docs) for doc_id in result_list]
            sessions.append(session)

        # Initialize parameters to neutral values
        A = defaultdict(lambda: 0.5)  # Attractiveness: initial guess
        S = defaultdict(lambda: 0.5)  # Satisfaction: initial guess

        # Train the model
        self.attractiveness, self.satisfaction = self._em(sessions, A, S)

    def simulate_clicks(self, qid, results):
        """
        Simulate user clicks using the DBN model for a given query and ranking.

        Parameters:
        -----------
        qid : str
            The query ID.

        results : list of str
            Ranked list of document IDs.

        Returns:
        --------
        clicks : list of int
            Simulated clicks (1 = click, 0 = no click) per document in the result list.
        """
        clicks = []
        examination = True  # User always examines the first result

        for doc_id in results:
            if not examination:
                clicks.append(0)
                continue

            # Get attractiveness and satisfaction; use defaults if missing
            A_qd = self.attractiveness.get((qid, doc_id), 0.0)
            S_qd = self.satisfaction.get((qid, doc_id), 0.5)

            click = 1 if random.random() < A_qd else 0
            clicks.append(click)

            if click:
                # If clicked: stop with probability of satisfaction
                examination = random.random() >= S_qd
            else:
                # If not clicked: continue
                examination = True

        return clicks


class PBM:
    """
    PBM (Position-Based Model) Click Model.

    This probabilistic model assumes that the probability of a click is
    the product of two independent factors:
        - Document attractiveness (depends on the query and document)
        - Position bias (depends only on the document rank)

    Attributes:
    -----------
    click_logs : pd.DataFrame
        Raw click logs.

    attractiveness : dict
        Estimated attractiveness values for (query_id, document_id) pairs.

    position_bias : dict
        Estimated position bias values for each rank.
    """

    def __init__(self, click_logs_path):
        """
        Initialize the PBM model.

        Parameters:
        -----------
        click_logs_path : str
            Path to the CSV file containing click log data.
        """
        self.click_logs = pd.read_csv(click_logs_path)
        self.attractiveness = None 
        self.position_bias = None

    def _em(self, sessions, A, G, max_iter=10):
        """
        Expectation-Maximization algorithm to estimate PBM parameters.

        Parameters:
        -----------
        sessions : list of list of tuples
            Each session is a list of (query_id, doc_id, rank, click) entries.

        A : dict
            Initial estimates of attractiveness values.

        G : dict
            Initial estimates of position bias values.

        max_iter : int
            Number of EM iterations.

        Returns:
        --------
        A, G : tuple of dicts
            Updated attractiveness and position bias dictionaries.
        """
        for iteration in range(max_iter):
            A_num = defaultdict(float)
            A_den = defaultdict(float)
            G_num = defaultdict(float)
            G_den = defaultdict(float)

            for session in sessions:
                for query_id, doc_id, rank, click in session:
                    key = (query_id, doc_id)
                    g_r = G[rank]
                    a_qd = A[key]

                    # Expected contribution of attractiveness and position bias
                    A_num[key] += click
                    A_den[key] += g_r
                    G_num[rank] += click
                    G_den[rank] += a_qd

            # M-step: Update A and G
            for key in A_den:
                A[key] = A_num[key] / A_den[key] if A_den[key] > 0 else A[key]
            for rank in G_den:
                G[rank] = G_num[rank] / G_den[rank] if G_den[rank] > 0 else G[rank]

        return A, G

    def update(self):
        """
        Process the click logs and estimate the model parameters
        (attractiveness and position bias) using EM algorithm.
        """
        grouped = self.click_logs.groupby(['user_id', 'query_id'])
        sessions = []

        for (user_id, query_id), group in grouped:
            result_list = group['query_results'].iloc[0].split(',')
            clicked_docs = set(group['clicked_document_id'].tolist())

            session = []
            for rank, doc_id in enumerate(result_list):
                click = int(doc_id in clicked_docs)
                session.append((query_id, doc_id, rank, click))
            sessions.append(session)

        # Initialize with default values
        A = defaultdict(lambda: 0.5)  # Attractiveness per (query_id, document_id)
        G = defaultdict(lambda: 0.9)  # Position bias per rank

        self.attractiveness, self.position_bias = self._em(sessions, A, G)

    def simulate_clicks(self, qid, results):
        """
        Simulate clicks for a given query and ranked result list.

        Parameters:
        -----------
        qid : str
            The query ID.

        results : list of str
            Ranked list of document IDs.

        Returns:
        --------
        clicks : list of int
            Simulated clicks (1 = click, 0 = no click) for each document in the list.
        """
        clicks = []
        for rank, doc_id in enumerate(results):
            a_qd = self.attractiveness.get((qid, doc_id), 0.2)  # default if unseen
            g_r = self.position_bias.get(rank, 0.9)             # default if unseen
            click_prob = a_qd * g_r

            click = 1 if random.random() < click_prob else 0
            clicks.append(click)
        return clicks


def main():
    click_logs_path='./data/logs/sampled/confirm_choose_logs_sampled_1.csv'
    qrels_path='./data/qrels/2022.qrels.pass.withDupes.txt'
    qid = 2027130
    results = ["msmarco_passage_05_837259438",
                "msmarco_passage_00_702052494",
                "msmarco_passage_66_11162916",
                "msmarco_passage_43_353320329",
                "msmarco_passage_41_185345572"]

    # Simulate clicks with DCTR
    dctr = DCTR(click_logs_path=click_logs_path)
    dctr.update()
    clicks = dctr.simulate_clicks(qid, results)
    print(clicks)

    # Simulate clicks with DCM 
    dcm = DCM(page_length=5, 
              click_logs_path=click_logs_path, 
              qrels_path=qrels_path)
    dcm.update_continuation()
    dcm.update_relevance()
    clicks = dcm.simulate_clicks(qid, results)
    print(clicks)

    # Simulate clicks with DBN 
    dbn = DBN(click_logs_path=click_logs_path)
    dbn.update()
    clicks = dbn.simulate_clicks(qid, results)
    print(clicks)

    # Simulate clicks with PBM 
    pbm = PBM(click_logs_path=click_logs_path)
    pbm.update()
    clicks = pbm.simulate_clicks(qid, results)
    print(clicks)

if __name__ == '__main__':
    main()
