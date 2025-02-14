import json 
import shelve 
from openai import OpenAI 

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

with shelve.open('../results/passage_title_store') as db:
    missing_passages = (set(passage_store.keys()).difference(set(dict(db).keys())))

passages = json.dumps({passageid: passage_store.get(passageid) for passageid in missing_passages})
prompt = "Generate a title for the following passages: {}. Provide the output as a Python dictionary without verbatim formatting.".format(passages) 

completion = client.chat.completions.create(
  model="gpt-4o",
  temperature=0.0,
  seed=42,
  messages=[
    {"role": "user", "content": prompt}
  ]
)
output = json.loads(completion.choices[0].message.content)
with shelve.open('../results/passage_title_store') as db:
    for passageid, title in output.items():
        db[passageid] = title