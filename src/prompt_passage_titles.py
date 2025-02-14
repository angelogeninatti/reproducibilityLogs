import json 
import shelve 
from itertools import islice
from openai import OpenAI 


def dict_chunks(data, chunk_size):
    """Yield successive chunk_size chunks from data."""
    it = iter(data)
    for i in range(0, len(data), chunk_size):
        yield {k: data[k] for k in islice(it, chunk_size)}


def main():
    chunk_size = 100
    client = OpenAI()

    passage_store = {}

    B = [round(0.2 + 0.05 * i, 3) for i in range(0,13)]
    K1 = [round(0.5 + 0.05 * i, 3) for i in range(0,13)]
    # BM25
    for b in B:
        for k1 in K1:
            run_name = '_'.join(['../runs/bm25/jsonl/bm25', str(b), str(k1)])
            file_name = '.'.join([run_name, 'jsonl'])
            with open(file_name) as f_in:
                lines = f_in.readlines()
                for line in lines:
                    p = json.loads(line)
                    passage_store[p.get('docno')] = p.get('text')
    # monoT5
    for bs in [4, 10]:
        for b in B:
            for k1 in K1:
                run_name = '../runs/monoT5/jsonl/monoT5_{}_bm25_{}_{}'.format(str(bs), str(b), str(k1))
                file_name = '.'.join([run_name, 'jsonl'])
                with open(file_name) as f_in:
                    lines = f_in.readlines()
                    for line in lines:
                        p = json.loads(line)
                        passage_store[p.get('docno')] = p.get('text')

    i = 1
    for chunk in dict_chunks(passage_store, chunk_size):
        print('Processing chunk no. {}'.format(str(i)))
        passages = json.dumps(dict(list(chunk.items())))
        prompt = "Generate a title for the following passages: {}. Provide the output as a Python dictionary without verbatim formatting.".format(passages) 
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            seed=42,
            messages=[{"role": "user", "content": prompt}]
            )
        output = json.loads(completion.choices[0].message.content)
        with shelve.open('../results/passage_title_store') as db:
            for passageid, title in output.items():
                db[passageid] = title
        i += 1

if __name__ == '__main__':
    main()
