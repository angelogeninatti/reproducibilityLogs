import json
import pandas as pd

def load_as_df(file_path):
    with open(file_path) as f_in:
        lines = [json.loads(line) for line in f_in.readlines()]
        run = pd.DataFrame(lines)
    return run

run_df = load_as_df('./runs/bm25/jsonl/bm25_0.2_0.5.jsonl')
qids = run_df.qid.unique()
qstrs = run_df.qstr.unique()

json_out = {}

for qid, qstr in zip(qids, qstrs):
    json_out[qid] = {'qstr': qstr}
    B = [round(0.2 + 0.05 * i, 3) for i in range(0,13)]
    K1 = [round(0.5 + 0.05 * i, 3) for i in range(0,13)]
    for b in B:
        for k1 in K1:
            run_name = '_'.join(['./runs/bm25/jsonl/bm25', str(b), str(k1)])
            file_name = '.'.join([run_name, 'jsonl'])
            run_df = load_as_df(file_name)
            ranking_df = run_df[run_df['qid'] == qid]
            ktu = ranking_df['ktu'].unique()[0]
            P_5 = ranking_df['P_5'].unique()[0]
            ndcg_cut_5 = ranking_df['ndcg_cut_5'].unique()[0]

            docnos = ranking_df['docno']
            ranks = ranking_df['rank']
            scores = ranking_df['score']
            urls = ranking_df['url']
            msmarco_document_ids = ranking_df['msmarco_document_id']
            texts = ranking_df['text']
            titles = ranking_df['title']
            titles_gpt4o = ranking_df['title_gpt4o']

            ranking = [{
                'docno': docno,
                'rank': rank,
                'score': score,
                'url': url,
                'msmarco_document_id': msmarco_document_id,
                'text': text,
                'title': title,
                'title_gpt4o': title_gpt4o
            } for docno, rank, score, url, msmarco_document_id, text, title, title_gpt4o in zip(docnos, ranks, scores, urls, msmarco_document_ids, texts, titles, titles_gpt4o)]
            json_out[qid][str((b, k1))] = {'ranking': ranking, 'ktu': ktu, 'P_5': P_5, 'ndcg_cut_5': ndcg_cut_5}

    for bs in [4, 10, 50]:
        for b in B:
            for k1 in K1:
                run_name = './runs/monoT5/jsonl/monoT5_{}_bm25_{}_{}'.format(str(bs), str(b), str(k1))
                file_name = '.'.join([run_name, 'jsonl'])
                run_df = load_as_df(file_name)
                ranking_df = run_df[run_df['qid'] == qid]
                ktu = ranking_df['ktu'].unique()[0]
                P_5 = ranking_df['P_5'].unique()[0]
                ndcg_cut_5 = ranking_df['ndcg_cut_5'].unique()[0]

                docnos = ranking_df['docno']
                ranks = ranking_df['rank']
                scores = ranking_df['score']
                urls = ranking_df['url']
                msmarco_document_ids = ranking_df['msmarco_document_id']
                texts = ranking_df['text']
                titles = ranking_df['title']
                titles_gpt4o = ranking_df['title_gpt4o']

                ranking = [{
                    'docno': docno,
                    'rank': rank,
                    'score': score,
                    'url': url,
                    'msmarco_document_id': msmarco_document_id,
                    'text': text,
                    'title': title,
                    'title_gpt4o': title_gpt4o
                } for docno, rank, score, url, msmarco_document_id, text, title, title_gpt4o in zip(docnos, ranks, scores, urls, msmarco_document_ids, texts, titles, titles_gpt4o)]
                json_out[qid][str((bs, b, k1))] = {'ranking': ranking, 'ktu': ktu, 'P_5': P_5, 'ndcg_cut_5': ndcg_cut_5}
    
with open('./rankings.json', 'w') as f_out:
    f_out.write(json.dumps(json_out, indent=4))
