# Now that your System has been Reproduced, What does this Mean for the Users?

## Prerequisites

Required Python packages can be installed via:
```bash
pip install -r requirements.txt
```
The code was developed using Python 3.12.

## Running the scripts

The repository includes both raw data and intermediate computational results. Researchers may either:
1. Use the provided intermediate files to reproduce specific analysis steps. In this case, no further action is required and the analysis may be run via:
   1. `python3 base_analysis.py` to run the analysis on the user study data, followed by `python3 merge.py` to generate cleaner outputs from the base analysis. This will result in the generation of `results/merged_questionnaire_results.csv` and `results/merged_statistics_*_*.csv`, which contain the final aggregated results of the analysis. Additionally, run `python3 relevance_statistics.py` to see the percentage of chosen items which were relevant according to the TREC dataset.
   2. `python3 sample_based_simulations.py` to simulate clicks with models trained on different data splits. The simulated clicks can be found in `simulations/`.
   3. `python3 ProgressiveAnalysis.py` to run the progressive analysis on the combined data (both from real and simulated users). This will result in the generation of `simulation/averaged_results` containing the final aggregated results of the progressive analysis.
2. Replicate the complete pipeline from raw data. You can run the entire pipeline from scratch by deleting all the intermediate files, or selectively delete specific files to re-run only specific parts of the analysis. The following steps provide a guide for this approach:
  -  **Using raw data**: if you prefer to use the raw experimental data instead of the snapshot, you can upload `user_study.csv` to a MySQL database, and set the connection parameters in `connection_data.py`. Then, delete the database snapshot (`df_test_log.bin`) and the pre-processed data for the analysis (`processed_logs.pkl`).
  -  **Removing the processed data of the base analysis**: if you prefer to perform the base analysis from scratch, delete `processed_logs.pkl` and run `python3 base_analysis.py`.
  -  **Generating sampled user data from scratch**: if you prefer to generate new samples from the user study data, delete the `sampled` folder and run `python3 sample_dataset.py`.
  -  **Removing the processed data of the progressive analysis**: if you prefer to run the progressive analysis from scratch, delete the `simulation/multi_runs`. Please note that this may take several hours to complete, as it requires running the analysis multiple times on multiple large datasets.
  -  Proceed with the execution of the complete pipeline as described in the previous point (1.i, 1.ii, 1.iii).

## Implementation Structure
Below is a more detailed description of the main modules in the repository. If you only need to run the analysis, you may refer to the previous section for a quick guide.

### Steps to reproduce the rankings for the online experiments

To recreate the rankings used in the online experiments, you have to follow the four basic steps of  1) indexing, 2) retrieval, 3) passage title generation, 4) offline evaluation. In the following, we outline what scripts need to be run to reproduce the steps. 

#### 1. Index creation
```
python src/create_docstore_index.py
```

#### 2. Passage retrieval with BM25 and monoT5

To retrieve the BM25 first-stage rankings run: 
```
python src/create_bm25_runs.py
```

To rerank the passages in the second stage, we used monoT5 that was run on Google Colab. To reproduce this step, use the notebook `src/monoT5_reranking.ipynb`. 

Finally, merge the rankings in the generated run files in a JSONL file for the later steps.
```
python src/create_jsonl.py
```

#### 3. Passage title generation
To generate titles for single passages, we used OpenAI's GPT-4o. Note that an API key is required to rerun the script.

```
python src/prompt_passage_titles.py &&
python src/add_titles_to_jsonl.py
```

#### 4. Evaluation of offline effectiveness and reproducibility
Finally, the offline (reproducibility) scores need to be added to the generated JSONL file. 

```
python src/add_scores_to_jsonl.py
```

In the end, there should be a file named `rankings.json` that is the basis for the online experiments.

To reproduce the table of the offline reproducibilty evaluation in the paper, run the following: 
```
python src/offline_reproducibility.py
```

### Data Preprocessing
### Base analysis
The analysis pipeline generates insights from the user study data. It consists of the following modules:
#### `base_analysis.py`
Primary analysis implementation for the user study data. It generates the following outputs:
- `basic_metrics.csv`: mean and standard deviation for all metrics across conditions
- `pvalue_table_*.csv`: p-values for the t-test for all metrics across conditions, both for the choose and expand signals.
- `questionnaire_basic.csv`: mean and standard deviation for all questionnaire responses across conditions
- `tost_tests_*.csv`: p-values for the TOST test for all metrics across conditions, both for the choose and expand signals.

Execution options:
- Keep `processed_logs.pkl` to only perform statistical analysis
- Execute complete analysis pipeline deleting `processed_logs.pkl` and using either:
  - Raw experimental data to be uploaded on a mysql database (`user_study.csv`)
  - Database snapshot (`df_test_log.bin`)

#### `merge.py`
Module to generate cleaner outputs from the base analysis after it is run:
- `merged_questionnaire_results.csv`: merged questionnaire results
- `merged_statistics_*_*.csv`: merged results with averages, standard deviations, and statistical significance, both for the choose and expand signals.

### `relevance_statistics.py`
Module to compare the relevance of the documents in the user study with the relevance of the documents in the TREC qrels file.

### Generating simulated clicks
#### `sample_dataset.py`
Generates random samples from the user study dataset. Researchers may:
- Utilize existing sampling outputs in `sampled` - in this case, running this script is not necessary.
- Generate new samples using either:
  - Database snapshot (`df_test_log.bin`)
  - Raw experimental data

### Progressive analysis
#### `ProgressiveAnalysis.py`
Implementation of progressive analysis methodology, combining data from the sampled versions of the user study and the generated clicks.
Input:
- `simulation` folder containing the sampled versions of the user study

Output in `simulation/averaged_results` for each metric and signal (either choose or expand):
- `*_*_averaged_diff.png` evolution of the difference between the base condition (with no generated data) and each metric/condition when progressively adding more simulated data.
- `*_*_t_test_averaged_results.png` and `*_*_tost_averaged_results.png` evolution of the statistical test results between each condition and the original one.
- `*_*_welch_vs_base_averaged_results.png` evolution of the Welch statistical test results between each condition and the original one.

Execution options:
- Keep `simulations/multi_runs` to only generate the analysis from the cached data
- Delete it to run the complete pipeline (this may take several hours)

### Auxiliary Modules
Additional Python modules contain implementation utilities and are not intended for direct execution.

### Source Data
- `user_study.csv`: Raw experimental data
- `df_test_log.bin`: Database state snapshot
- `2022.qrels.pass.withDupes.txt`: TREC qrels file (required only for `relevance_statistics.py`, if not using pre-computed mappings)

### Rankings
The generated rankings can be found in a separate data repository on Zenodo: https://zenodo.org/records/14886328

### Web application for the user study
The code for the web application is available at https://github.com/angelogeninatti/reproducibility-UI
