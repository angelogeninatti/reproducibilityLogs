import pandas as pd
import numpy as np
from pathlib import Path


def create_samples(input_file, output_dir, n_iterations=5, min_instances=2):
    """
    Create stratified random samples ensuring minimum instances per group.

    Args:
        input_file (str): Path to input CSV file
        output_dir (str): Directory to save sampled datasets
        n_iterations (int): Number of sampling iterations
        min_instances (int): Minimum instances required per group
    """
    # Read the data
    df = pd.read_csv(input_file)

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Function to perform stratified sampling while maintaining minimum instances
    def stratified_sample(data, sample_frac=0.5):
        # Create groups based on the required conditions
        groups = data.groupby(['query_id', 'condition_b', 'condition_k1'])

        sampled_dfs = []

        for name, group in groups:
            group_size = len(group)
            # Calculate sample size for this group (at least min_instances)
            sample_size = max(min_instances, int(np.ceil(group_size * sample_frac)))
            # If group is smaller than min_instances, take all rows
            if group_size <= min_instances:
                sampled_group = group
            else:
                # Random sample from the group
                sampled_group = group.sample(n=sample_size, random_state=None)
            sampled_dfs.append(sampled_group)

        # Combine all sampled groups
        return pd.concat(sampled_dfs, ignore_index=True)

    # Perform sampling for each iteration
    for i in range(n_iterations):
        # Get stratified sample
        sampled_df = stratified_sample(df)

        # Save to file
        output_file = f"{output_dir}/{input_file.split(".")[0]}_sampled_{i + 1}.csv"
        sampled_df.to_csv(output_file, index=False)

        # Print statistics
        print(f"\nIteration {i + 1}:")
        print(f"Total rows in sample: {len(sampled_df)}")
        print(f"Original rows: {len(df)}")
        print(f"Sampling ratio: {len(sampled_df) / len(df):.2%}")

        # Verify minimum instances requirement
        groups = sampled_df.groupby(['query_id', 'condition_b', 'condition_k1'])
        min_group_size = groups.size().min()
        print(f"Minimum instances per group: {min_group_size}")


if __name__ == "__main__":
    # Create samples
    create_samples(
        input_file="confirm_choose_logs.csv",
        output_dir="sampled",
        n_iterations=5,
        min_instances=2
    )
    create_samples(
        input_file="expand_result_logs.csv",
        output_dir="sampled",
        n_iterations=5,
        min_instances=2
    )