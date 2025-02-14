import os
import json
import ir_datasets
import pyterrier as pt


def main():
    if not pt.started():
        pt.init()

    dataset_document = ir_datasets.load("msmarco-document-v2")
    dataset_passage = ir_datasets.load("msmarco-passage-v2")
    docstore_document = dataset_document.docs_store()
    docstore_passage = dataset_passage.docs_store()

    dataset_passage = pt.get_dataset('irds:msmarco-passage-v2/trec-dl-2022/judged')
    queries_df = dataset_passage.get_topics()
    queries = {}
    for row in queries_df.iterrows():
        queries[row[1].qid] = row[1].query 

    B = [round(0.2 + 0.05 * i, 3) for i in range(0,13)]
    K1 = [round(0.5 + 0.05 * i, 3) for i in range(0,13)]

    # BM25 runs
    for b in B:
        for k1 in K1:
            run_name = '_'.join(['bm25', str(b), str(k1)])
            file_name_in = '.'.join([run_name, 'txt'])
            run = pt.io.read_results(os.path.join('runs', 'bm25', 'txt', file_name_in))
            file_name_out = '.'.join([run_name, 'jsonl'])
            file_path_out = os.path.join('runs', 'bm25', 'jsonl', file_name_out)

            with open(file_path_out, 'w') as f_out:
                for ranking in run.groupby('qid'):
                    if ranking[0] in queries.keys():
                        for row in list(ranking[1].iterrows())[:10]:
                            passage = docstore_passage.get(row[1].docno)
                            text = passage.text 
                            msmarco_document_id = passage.msmarco_document_id
                            document = docstore_document.get(msmarco_document_id)
                            title = document.title
                            url = document.url
                            f_out.write(json.dumps({
                                'qid': row[1].qid,
                                'qstr': queries[row[1].qid],
                                'docno': row[1].docno,
                                'rank': row[1]['rank'],
                                'score': row[1].score,
                                'runid': row[1].name,
                                'url': url,
                                'msmarco_document_id': msmarco_document_id,
                                'text': text, 
                                'title': title
                            }) + '\n')

    # monoT5 runs
    for bs in [4, 10, 50]:
        for b in B:
            for k1 in K1:
                run_name = '_'.join(['monoT5', str(bs), 'bm25', str(b), str(k1)])
                file_name_in = '.'.join([run_name, 'txt'])
                run = pt.io.read_results(os.path.join('runs', 'monoT5', 'txt', file_name_in))
                file_name_out = '.'.join([run_name, 'jsonl'])
                file_path_out = os.path.join('runs', 'monoT5', 'jsonl', file_name_out)

                with open(file_path_out, 'w') as f_out:
                    for ranking in run.groupby('qid'):
                        if ranking[0] in queries.keys():
                            for row in list(ranking[1].iterrows())[:10]:
                                passage = docstore_passage.get(row[1].docno)
                                text = passage.text 
                                msmarco_document_id = passage.msmarco_document_id
                                document = docstore_document.get(msmarco_document_id)
                                title = document.title
                                url = document.url
                                f_out.write(json.dumps({
                                    'qid': row[1].qid,
                                    'qstr': queries[row[1]['qid']],
                                    'docno': row[1]['docno'],
                                    'rank': row[1]['rank'],
                                    'score': row[1]['score'],
                                    'runid': row[1]['name'],
                                    'url': url,
                                    'msmarco_document_id': msmarco_document_id,
                                    'text': text, 
                                    'title': title
                                }) + '\n')


if __name__ == '__main__':
    main()
