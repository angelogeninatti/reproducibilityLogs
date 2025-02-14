import pandas as pd
import numpy as np

CONDITIONS = {
    0: {'b': 0.5, 'k1': 0.8},
    1: {'b': 0.2, 'k1': 0.5},
    2: {'b': 0.25, 'k1': 0.6},
    3: {'b': 0.65, 'k1': 1.1},
    4: {'b': 0.75, 'k1': 0.85},
    5: {'b': 0.8, 'k1': 0.95},
    7: {'b': 0.2, 'k1': 0.85}  # Added new condition
}


def get_condition_label(condition):
    params = CONDITIONS[condition]
    return f"({params['b']},{params['k1']})"


def get_condition_label_noBase(condition):
    next_condition = condition + 1
    if next_condition == 6:  # Skip index 6
        next_condition = 7
    params = CONDITIONS[next_condition]
    return f"{params['b']},{params['k1']}"


def merge_statistical_data(basic_metrics_path, statistical_tests_pattern, tost_tests_pattern):
    basic_df = pd.read_csv(basic_metrics_path)
    click_types = basic_df['click_type'].unique()
    results = {}

    for click_type in click_types:
        click_metrics = basic_df[basic_df['click_type'] == click_type]
        stat_tests = pd.read_csv(statistical_tests_pattern.format(click_type))
        tost_tests = pd.read_csv(tost_tests_pattern.format(click_type))
        metrics = click_metrics['metric'].unique()

        # Updated to include condition 7
        result_data = {
            'condition': [get_condition_label(i) for i in [0, 1, 2, 3, 4, 5, 7]]
        }

        for metric in metrics:
            result_data[metric] = [None] * 7  # Changed to 7 elements

            for idx, condition in enumerate([0, 1, 2, 3, 4, 5, 7]):  # Modified iteration
                metric_basic = click_metrics[(click_metrics['metric'] == metric) &
                                             (click_metrics['condition'] == condition)]
                metric_stat = stat_tests[(stat_tests['metric'] == metric) &
                                         (stat_tests['comparison_condition'] == condition)]
                metric_tost = tost_tests[(tost_tests['metric'] == metric) &
                                         (tost_tests['comparison_condition'] == condition)]

                mean_val = metric_basic['mean'].values[0] if not metric_basic.empty else np.nan
                std_val = metric_basic['std'].values[0] if not metric_basic.empty else np.nan

                if condition == 0:
                    formatted_content = f"{mean_val:.3f} ({std_val:.3f})"
                else:
                    t_test_p_val = metric_stat['t_test_p_value'].values[0] if not metric_stat.empty else np.nan
                    tost_p_val = metric_tost['classic_tost_p_value'].values[0] if not metric_tost.empty else np.nan
                    formatted_content = f"{mean_val:.3f} ({std_val:.3f})\n{t_test_p_val:.3f}, {tost_p_val:.3f}"

                result_data[metric][idx] = formatted_content

        result_df = pd.DataFrame(result_data)
        output_file = f'results/merged_statistics_{click_type}.csv'
        result_df.to_csv(output_file, index=False)
        results[click_type] = result_df

    return results


def merge_questionnaire_data():
    basic_df = pd.read_csv('results/questionnaire_basic.csv')
    stats_df = pd.read_csv('results/questionnaire_statistical_tests.csv')
    questions = basic_df['question'].unique()
    result_data = []

    for question in questions:
        row_data = {'question': question}

        # Reference condition
        ref_data = basic_df[(basic_df['question'] == question) &
                            (basic_df['condition'] == 0)].iloc[0]
        row_data[get_condition_label(0)] = f"{ref_data['mean']} ({ref_data['std']})"

        # Other conditions, including condition 7
        for condition in [1, 2, 3, 4, 5, 7]:  # Modified iteration
            condition_data = basic_df[(basic_df['question'] == question) &
                                      (basic_df['condition'] == condition)]

            if not condition_data.empty:
                condition_data = condition_data.iloc[0]
                stats_data = stats_df[(stats_df['question'] == question) &
                                      (stats_df['comparison_condition'] == condition)]

                if not stats_data.empty:
                    stats_data = stats_data.iloc[0]
                    row_data[get_condition_label(
                        condition)] = f"{condition_data['mean']} ({condition_data['std']})\n{stats_data['t_test_p_value']},{stats_data['classic_tost_p_value']}"
                else:
                    row_data[get_condition_label(condition)] = f"{condition_data['mean']} ({condition_data['std']})"

        result_data.append(row_data)

    return pd.DataFrame(result_data)


if __name__ == "__main__":
    result = merge_statistical_data(
        'results/basic_metrics.csv',
        'results/statistical_tests_{}.csv',
        'results/tost_tests_{}.csv'
    )
    result_df = merge_questionnaire_data()
    result_df.to_csv('results/merged_questionnaire_results.csv', index=False)