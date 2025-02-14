import json
import shelve

B = [round(0.2 + 0.05 * i, 3) for i in range(0,13)]
K1 = [round(0.5 + 0.05 * i, 3) for i in range(0,13)]

with shelve.open('./results/passage_title_store') as db:
    # BM25
    for b in B:
        for k1 in K1:
            run_name = '_'.join(['./runs/bm25/jsonl/bm25', str(b), str(k1)])
            file_name = '.'.join([run_name, 'jsonl'])
            run_out = '_'.join(['./runs/bm25/jsonl/update/bm25', str(b), str(k1)])
            output_name = '.'.join([run_out, 'jsonl'])
            with open(file_name, 'r') as input_file, open(output_name, 'w') as output_file:
                for line in input_file:
                    data = json.loads(line)
                    data['title_gpt4o'] = db.get(data.get('docno'))
                    json.dump(data, output_file)
                    output_file.write('\n')  
    # monoT5
    for bs in [4, 10, 50]:
        for b in B:
            for k1 in K1:
                run_name = './runs/monoT5/jsonl/monoT5_{}_bm25_{}_{}'.format(str(bs), str(b), str(k1))
                file_name = '.'.join([run_name, 'jsonl'])
                run_out = './runs/monoT5/jsonl/update/monoT5_{}_bm25_{}_{}'.format(str(bs), str(b), str(k1))
                output_name = '.'.join([run_out, 'jsonl'])
                with open(file_name, 'r') as input_file, open(output_name, 'w') as output_file:
                    for line in input_file:
                        data = json.loads(line)
                        data['title_gpt4o'] = db.get(data.get('docno'))
                        json.dump(data, output_file)
                        output_file.write('\n') 