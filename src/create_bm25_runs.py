import os
from tqdm import tqdm 
import pyterrier as pt


def main():
    if not pt.started():
        pt.init()
    
    dataset_passage = pt.get_dataset('irds:msmarco-passage-v2/trec-dl-2022')
    index_ref_passage = pt.IndexRef.of('./indices/msmarco-passage-v2_dedup')

    B = [round(0.2 + 0.05 * i, 3) for i in range(0, 13)]
    K1 = [round(0.5 + 0.05 * i, 3) for i in range(0, 13)]

    T = 1000

    total = len(B)*len(K1)

    bm25 = pt.BatchRetrieve(index_ref_passage , wmodel='BM25', controls={"bm25.b" : 0.75, "bm25.k_1": 0.75, "bm25.k_3": 0.75}, num_results=T)
    with tqdm(total=total, position=0, leave=True) as pbar:
        for b in B:
            for k1 in K1:
                bm25.set_parameter('bm25.b', b)
                bm25.set_parameter('bm25.k_1', k1)
                res = bm25.transform(dataset_passage.get_topics('text'))
                run_name = '_'.join(['bm25', str(b), str(k1)])
                file_name = '.'.join([run_name, 'txt'])
                pt.io.write_results(res, os.path.join('runs', 'bm25', 'txt', file_name), format='trec',run_name=run_name)
                pbar.update()


if __name__ == '__main__':
    main()
