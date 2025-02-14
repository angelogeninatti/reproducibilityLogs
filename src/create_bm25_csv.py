import os 
from tqdm import tqdm
import pandas as pd
import ir_datasets
import pyterrier as pt


def main():
    if not pt.started():
        pt.init()

    dataset_passage = ir_datasets.load("msmarco-passage-v2")
    docstore_passage = dataset_passage.docs_store()

    dataset_passage = pt.get_dataset('irds:msmarco-passage-v2/trec-dl-2022/judged')
    queries_df = dataset_passage.get_topics()
    queries = {}
    for row in queries_df.iterrows():
        queries[row[1].qid] = row[1].query 

    B = [round(0.2 + 0.05 * i, 3) for i in range(0, 13)]
    K1 = [round(0.5 + 0.05 * i, 3) for i in range(0, 13)]

    total = len(B)*len(K1)

    with tqdm(total=total, position=0, leave=True) as pbar:
        for b in B:
            for k1 in K1:
                run_name = '_'.join(['bm25', str(b), str(k1)])
                file_name_in = '.'.join([run_name, 'tar', 'gz'])
                run = pt.io.read_results(os.path.join('runs', 'bm25', 'compressed', file_name_in))

                _qid = []
                _query = []
                _docno = []
                _text = []

                for qid in queries_df['qid']:
                    _qid_tmp = list(run[run['qid'] == qid]['qid'])
                    _qid += _qid_tmp
                    _query += list(pd.Series(queries.get(q) for q in _qid_tmp))
                    _docno += list(run[run['qid'] == qid]['docno'])
                    _text += list(pd.Series([docstore_passage.get(docno).text for docno in run[run['qid'] == qid]['docno']]))

                run_df = pd.DataFrame([
                    [__qid, __query, __docno, __text] for __qid, __query, __docno, __text in zip(_qid,_query,_docno,_text)
                ], columns=['qid', 'query', 'docno', 'text'])

                file_name_out = '.'.join([run_name, 'csv'])
                run_df.to_csv(os.path.join('runs', 'bm25', 'csv', file_name_out), index=False)
                pbar.update()


if __name__ == '__main__':
    main()