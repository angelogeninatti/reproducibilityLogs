import pickle
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from CSVLogAdapter import CSVLogAdapter
from base_analysis import process_logs, tost_test
from logs import Logs


@dataclass
class ProcessingConfig:
    """Configuration for parallel processing"""
    simulation_dir: Path
    y_value: int
    last_percentage: float
    current_percentage: float
    force_reload: bool


def get_condition_style(condition_str):
    """Get consistent color and style for a given condition string."""
    # Define condition parameters
    CONDITIONS = {
        0: {'b': 0.5, 'k1': 0.8},  # Base condition
        1: {'b': 0.2, 'k1': 0.5},
        2: {'b': 0.25, 'k1': 0.6},
        3: {'b': 0.65, 'k1': 1.1},
        4: {'b': 0.75, 'k1': 0.85},
        5: {'b': 0.8, 'k1': 0.95},
        7: {'b': 0.2, 'k1': 0.85}
    }

    # Define colors for each condition ID using RGB values
    COLORS = {
        0: (255 / 255, 129 / 255, 23 / 255),  # Blue
        1: (0 / 255, 184 / 255, 241 / 255),  # Orange
        2: (129 / 255, 0 / 255, 128 / 255),  # Green
        3: (0 / 255, 0 / 255, 254 / 255),  # Red
        4: (0 / 255, 255 / 255, 4 / 255),  # Purple
        5: (193 / 255, 129 / 255, 66 / 255),  # Brown
        7: (255 / 255, 191 / 255, 192 / 255)  # Pink
    }

    # Parse condition string to match with CONDITIONS
    parts = condition_str.split(',')
    condition_dict = {}
    for part in parts:
        if '=' in part:
            key, value = part.split('=')
            condition_dict[key] = float(value)

    # Find matching condition ID
    for cid, params in CONDITIONS.items():
        if params['b'] == condition_dict.get('b') and params['k1'] == condition_dict.get('k1'):
            return COLORS[cid]

    return 'rgb(127, 127, 127)'  # Default gray for unmatched conditions


def save_standalone_legend():
    """Create and save a standalone legend with all conditions."""
    plt.figure(figsize=(6, 8))
    ax = plt.gca()

    # Hide the axes
    ax.set_axis_off()

    # Define conditions with their parameters
    CONDITIONS = {
        0: {'b': 0.5, 'k1': 0.8},  # Original system
        1: {'b': 0.2, 'k1': 0.5},
        2: {'b': 0.25, 'k1': 0.6},
        3: {'b': 0.65, 'k1': 1.1},
        4: {'b': 0.75, 'k1': 0.85},
        5: {'b': 0.8, 'k1': 0.95},
        7: {'b': 0.2, 'k1': 0.85}
    }

    # Define colors for each condition ID using RGB values
    COLORS = {
        0: (255 / 255, 129 / 255, 23 / 255),  # Blue
        1: (0 / 255, 184 / 255, 241 / 255),  # Orange
        2: (129 / 255, 0 / 255, 128 / 255),  # Green
        3: (0 / 255, 0 / 255, 254 / 255),  # Red
        4: (0 / 255, 255 / 255, 4 / 255),  # Purple
        5: (193 / 255, 129 / 255, 66 / 255),  # Brown
        7: (255 / 255, 191 / 255, 192 / 255)  # Pink
    }

    # Create dummy lines for each condition with updated labels
    for cid, params in sorted(CONDITIONS.items()):
        if cid == 0:
            label = "original"
        else:
            label = f"rep{cid}"

        plt.plot([], [],
                 label=label,
                 color=COLORS[cid],
                 linewidth=2.5,
                 marker='o',
                 markersize=10)

    # Add legend
    plt.legend(title='Systems',
               loc='center')

    # Save legend as PDF
    plt.savefig('legend.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def setup_plot_common():
    """Common setup for all plots."""
    plt.figure(figsize=(20, 15))
    ax = plt.gca()

    # Set up grid
    plt.grid(True, alpha=0.3)

    # Return axes for further customization
    return ax

def save_process_logs_results(results, cache_path: Path):
    """Save process_logs results to a pickle file"""

    # Convert defaultdict to regular dict recursively
    def convert_defaultdict(obj):
        if isinstance(obj, (dict, defaultdict)):
            return {k: convert_defaultdict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_defaultdict(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_defaultdict(item) for item in obj)
        return obj

    # Create cache directory if it doesn't exist
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert and save results
    converted_results = convert_defaultdict(results)
    with open(cache_path, 'wb') as f:
        pickle.dump(converted_results, f)
    print(f"Cached process_logs results to {cache_path}")


def process_y_value(config: ProcessingConfig) -> Tuple[dict, int]:
    """Standalone function for processing a single Y value"""
    combined_incremental_data = []

    # Get files for this Y value
    files = {
        'choose': {},
        'expand': {}
    }

    # Find relevant files
    for file in config.simulation_dir.glob('*.csv'):
        try:
            if file.name.startswith('confirm_choose_logs_sampled_'):
                y_value = int(file.stem.split('_')[-1])
                if y_value == config.y_value:
                    files['choose'][y_value] = file
            elif file.name.startswith('expand_logs_sampled_'):
                y_value = int(file.stem.split('_')[-1])
                if y_value == config.y_value:
                    files['expand'][y_value] = file
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse Y value from filename {file.name}: {e}")
            continue

    # Process choose and expand files
    for type_key in ['choose', 'expand']:
        if config.y_value in files[type_key]:
            df = pd.read_csv(files[type_key][config.y_value])

            # Get incremental data
            start_idx = int(len(df) * config.last_percentage)
            end_idx = int(len(df) * config.current_percentage)
            incremental_data = df.iloc[start_idx:end_idx]

            if not incremental_data.empty:
                combined_incremental_data.append(incremental_data)

    if not combined_incremental_data:
        return None, 0

    try:
        # Process the incremental data
        all_incremental = pd.concat(combined_incremental_data, ignore_index=True)

        # Create adapter and get logs
        csv_adapter = CSVLogAdapter(all_incremental)
        increment_logs = csv_adapter.get_logs()

        # Create Logs object for this increment
        increment_only = Logs(preprocessed_data=increment_logs)

        # Generate cache path
        cache_dir = config.simulation_dir / "cache"
        cache_filename = f"process_logs_y{config.y_value}_pct{config.last_percentage:.3f}_{config.current_percentage:.3f}.pkl"
        cache_path = cache_dir / cache_filename

        # Try to load from cache first
        if not config.force_reload and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    analysis_results = pickle.load(f)
                return analysis_results[0], sum(len(logs.get_logs("confirmed")) for _, logs in increment_logs.items())
            except Exception as e:
                print(f"Error loading cached results: {e}")

        # Process logs if not cached or cache loading failed
        analysis_results = process_logs(increment_only, force_reload=True)

        if analysis_results and analysis_results[0]:
            # Cache the results
            cache_dir.mkdir(parents=True, exist_ok=True)
            if analysis_results:
                save_process_logs_results(analysis_results, cache_path)
            print(f"Cached results for Y={config.y_value}")

            return analysis_results[0], sum(len(logs.get_logs("confirmed")) for _, logs in increment_logs.items())

    except Exception as e:
        print(f"Error processing increment for Y={config.y_value}: {e}")
        raise e

    return None, 0


class IncrementalProgressiveAnalysis:
    def __init__(self, simulation_dir: str, results_dir: str, results_augmented_dir: str):
        self.simulation_dir = Path(simulation_dir)
        self.results_dir = Path(results_dir)
        self.results_augmented_dir = Path(results_augmented_dir)

        # Verify directories
        if not self.simulation_dir.exists():
            raise ValueError(f"Simulation directory '{simulation_dir}' does not exist")

        # Create output directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_augmented_dir.mkdir(parents=True, exist_ok=True)

        # Store metrics and their weights
        self.base_metrics = None
        self.base_weight = 0  # Number of logs in base analysis
        self.progressive_metrics = {}  # Will store {percentage: (metrics, weight)}
        self.statistical_results = defaultdict(lambda: {
            'choose': defaultdict(dict),
            'expand': defaultdict(dict),
        })

    def _run_statistical_tests(self, metrics: dict, percentage: float):
        """Run statistical tests on metrics and store results"""
        base_condition = "b=0.5,k1=0.8,batch_size=4"

        for click_type in ['choose', 'expand']:
            # Check if click_type exists in base condition
            if not metrics.get(base_condition, {}).get(click_type):
                continue

            for metric_name in metrics[base_condition][click_type].keys():
                if 'time' in metric_name.lower():
                    continue

                # Get base values
                try:
                    base_values = metrics[base_condition][click_type][metric_name]
                    if isinstance(base_values, (int, float)):
                        base_values = [base_values]
                    elif isinstance(base_values, np.ndarray):
                        base_values = base_values.tolist()
                    elif isinstance(base_values, list):
                        base_values = [x for x in base_values if not np.isnan(x)]
                except Exception as e:
                    print(f"Error getting base values for {metric_name}: {e}")
                    continue

                # Get historical base values (from self.base_metrics) for Welch's test
                try:
                    historical_base_values = self.base_metrics[base_condition][click_type][metric_name]
                    if isinstance(historical_base_values, (int, float)):
                        historical_base_values = [historical_base_values]
                    elif isinstance(historical_base_values, np.ndarray):
                        historical_base_values = historical_base_values.tolist()
                    elif isinstance(historical_base_values, list):
                        historical_base_values = [x for x in historical_base_values if not np.isnan(x)]
                except Exception as e:
                    print(f"Error getting historical base values for {metric_name}: {e}")
                    historical_base_values = None

                for condition in metrics.keys():
                    if condition == base_condition:
                        continue

                    try:
                        comp_values = metrics[condition][click_type][metric_name]
                        if isinstance(comp_values, (int, float)):
                            comp_values = [comp_values]
                        elif isinstance(comp_values, np.ndarray):
                            comp_values = comp_values.tolist()
                        elif isinstance(comp_values, list):
                            comp_values = [x for x in comp_values if not np.isnan(x)]

                        # Only proceed if we have enough values
                        if len(base_values) < 2 or len(comp_values) < 2:
                            print(f"  Skipping: insufficient data points")
                            continue

                        # Regular t-test between conditions
                        t_stat, t_p_value = stats.ttest_ind(base_values, comp_values, equal_var=False)

                        # Welch's test between current and historical base values
                        if historical_base_values and len(historical_base_values) >= 2:
                            welch_stat, welch_p_value = stats.ttest_ind(
                                comp_values,
                                historical_base_values,
                                equal_var=False  # This makes it Welch's t-test
                            )
                        else:
                            welch_p_value = np.nan

                    except Exception as e:
                        print(f"Error processing comparison values for {metric_name}, {click_type}, {condition}: {e}")
                        continue

                    # TOST test
                    try:
                        pooled_std = np.std(np.concatenate([base_values, comp_values]))
                        epsilon = 0.25 * pooled_std
                        tost_p_value = tost_test(base_values, comp_values, epsilon)
                    except Exception as e:
                        print(f"Error in TOST test for {metric_name}, {click_type}, {condition}: {e}")
                        tost_p_value = np.nan

                    # Store results
                    self.statistical_results[percentage][click_type][metric_name][condition] = {
                        't_test': t_p_value,
                        'tost': tost_p_value,
                        'welch_vs_base': welch_p_value
                    }

    def _get_cache_path(self, y: int, start_pct: float, end_pct: float) -> Path:
        """Generate a cache filepath for process_logs results"""
        cache_filename = f"process_logs_y{y}_pct{start_pct:.3f}_{end_pct:.3f}.pkl"
        return self.simulation_dir / "cache" / cache_filename

    def _load_process_logs_results(self, cache_path: Path):
        """Load process_logs results from a pickle file"""
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    def _get_simulation_files(self) -> Dict[str, Dict[int, Path]]:
        """Get simulation files grouped by type and Y value"""
        files = {
            'choose': {},
            'expand': {}
        }

        for file in self.simulation_dir.glob('*.csv'):
            try:
                if file.name.startswith('confirm_choose_logs_sampled_'):
                    y_value = int(file.stem.split('_')[-1])
                    files['choose'][y_value] = file
                elif file.name.startswith('expand_logs_sampled_'):
                    y_value = int(file.stem.split('_')[-1])
                    files['expand'][y_value] = file
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse Y value from filename {file.name}: {e}")

        return files

    def run_base_analysis(self, db_logs):
        """Run analysis on base logs and store their weight"""
        print("Running base analysis...")

        # Get base metrics from pickle or analyze
        pickle_file = self.results_dir / 'base_metrics.pkl'
        if pickle_file.exists():
            f = open(pickle_file, 'rb')
            all_metrics = pickle.load(f)
            self.base_metrics = all_metrics
            f.close()
        else:
            all_metrics, *_ = process_logs(db_logs, force_reload=True)
            self.base_metrics = all_metrics
            # Save to pickle:
            with open(self.results_dir / 'base_metrics.pkl', 'wb') as f:
                pickle.dump(all_metrics, f)

        self._run_statistical_tests(self.base_metrics, 0.0)

        # Store the weight (number of logs)
        self.base_weight = 0
        for uid, logs in db_logs.logs.items():
            self.base_weight += len(logs.get_logs("confirmed"))

        return all_metrics

    def _get_incremental_sample(self, df: pd.DataFrame, start_pct: float, end_pct: float) -> pd.DataFrame:
        """Get only the incremental data between two percentages"""
        start_idx = int(len(df) * start_pct)
        end_idx = int(len(df) * end_pct)
        return df.iloc[start_idx:end_idx]

    def _weighted_average_metrics(self, metrics_list: List[dict], weights: List[int]) -> dict:
        """Compute weighted average of metrics"""
        if not metrics_list or not weights or len(metrics_list) != len(weights):
            raise ValueError("Invalid metrics or weights")

        # Initialize structure to hold summed weighted values and total weights
        weighted_sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        weight_sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        # Collect all possible keys
        conditions = set()
        click_types = set()
        metric_names = set()

        for metrics in metrics_list:
            for condition in metrics:
                conditions.add(condition)
                for click_type in metrics[condition]:
                    click_types.add(click_type)
                    for metric in metrics[condition][click_type]:
                        if 'time' not in metric.lower():
                            metric_names.add(metric)

        # Calculate weighted sums
        for metrics, weight in zip(metrics_list, weights):
            for condition in conditions:
                if condition not in metrics:
                    continue

                for click_type in click_types:
                    if click_type not in metrics[condition]:
                        continue

                    for metric in metric_names:
                        if metric not in metrics[condition][click_type]:
                            continue

                        value = metrics[condition][click_type][metric]
                        if isinstance(value, (list, np.ndarray)):
                            value = np.mean(value)

                        weighted_sums[condition][click_type][metric] += value * weight
                        weight_sums[condition][click_type][metric] += weight

        # Calculate weighted averages
        result = {}
        for condition in conditions:
            result[condition] = {}
            for click_type in click_types:
                result[condition][click_type] = {}
                for metric in metric_names:
                    total_weight = weight_sums[condition][click_type][metric]
                    if total_weight > 0:
                        result[condition][click_type][metric] = (
                                weighted_sums[condition][click_type][metric] / total_weight
                        )

        return result

    @staticmethod
    def combine_raw_metrics(metrics_list: List[dict]) -> dict:
        """
        Combine raw metrics from multiple sources while preserving array structure for statistical testing.

        Args:
            metrics_list: List of metrics dictionaries to combine

        Returns:
            Combined metrics dictionary with preserved arrays
        """
        combined = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # For each metrics dict in the list
        for metrics in metrics_list:
            # For each condition (e.g. "b=0.5,k1=0.8,batch_size=4")
            for condition in metrics:
                # For each click type (choose, expand)
                for click_type in metrics[condition]:
                    # For each metric (ctr@5, ndcg@5, etc.)
                    for metric_name, values in metrics[condition][click_type].items():
                        if 'time' in metric_name.lower():
                            continue

                        # If value is a single number, convert to list
                        if isinstance(values, (int, float)):
                            values = [values]
                        # If numpy array, convert to list
                        elif isinstance(values, np.ndarray):
                            values = values.tolist()
                        # If already a list, use as is
                        elif isinstance(values, list):
                            values = [x for x in values if not np.isnan(x)]

                        # Extend the existing list with new values
                        combined[condition][click_type][metric_name].extend(values)

        # Convert defaultdict to regular dict
        return {
            condition: {
                click_type: dict(metrics)
                for click_type, metrics in click_types.items()
            }
            for condition, click_types in combined.items()
        }

    def run_incremental_analysis(self, percentages: List[float], y_values: List[int], force_reload=False):
        """Run incremental analysis with progressively more data using true parallel processing"""
        print("Running incremental analysis...")

        # Sort percentages
        percentages = sorted(percentages)

        num_processes = min(int(cpu_count()), len(y_values))
        print(f"Using {num_processes} processes for parallel processing")

        last_percentage = 0
        for current_percentage in percentages:
            print(f"\n=== Processing increment {last_percentage * 100}% to {current_percentage * 100}% ===")

            # Create processing configurations for each Y value
            configs = [
                ProcessingConfig(
                    simulation_dir=self.simulation_dir,
                    y_value=y,
                    last_percentage=last_percentage,
                    current_percentage=current_percentage,
                    force_reload=force_reload
                )
                for y in y_values
            ]

            # Process Y values in parallel
            with Pool(processes=num_processes) as pool:
                results = pool.map(process_y_value, configs)

            # Filter out None results and separate metrics and weights
            valid_results = [(m, w) for m, w in results if m is not None]
            if valid_results:
                increment_metrics, increment_weights = zip(*valid_results)

                # Combine with previous metrics using weighted average
                all_metrics = [self.base_metrics] + list(increment_metrics)
                all_weights = [self.base_weight] + list(increment_weights)

                avg_metrics = self._weighted_average_metrics(all_metrics, all_weights)
                self.progressive_metrics[current_percentage] = (avg_metrics, sum(increment_weights))

                raw_combined = self.combine_raw_metrics([self.base_metrics] + list(increment_metrics))

                self._run_statistical_tests(raw_combined, current_percentage)

            last_percentage = current_percentage

    def _create_statistical_plot(self, click_type: str, metric: str):
        """Create separate plots showing p-values from t-test, TOST test, and Welch's test"""
        tests = ['t_test', 'tost', 'welch_vs_base']
        test_labels = {
            't_test': 't-test',
            'tost': 'TOST',
            'welch_vs_base': "Welch's t-test vs Base"
        }

        percentages = [0.0] + sorted(self.progressive_metrics.keys())
        base_condition = "b=0.5,k1=0.8,batch_size=4"

        if click_type not in self.statistical_results[0.0]:
            print(f"No statistical results for {click_type}")
            return

        if metric not in self.statistical_results[0.0][click_type]:
            print(f"No statistical results for {metric} in {click_type}")
            return

        conditions = set()
        for p in percentages:
            if p in self.statistical_results and click_type in self.statistical_results[p]:
                if metric in self.statistical_results[p][click_type]:
                    conditions.update(self.statistical_results[p][click_type][metric].keys())

        # Create separate plots for each test
        for test in tests:
            ax = setup_plot_common()

            for condition in sorted(conditions):
                if condition == base_condition:
                    continue

                test_values = []

                for p in percentages:
                    if (p in self.statistical_results and
                            click_type in self.statistical_results[p] and
                            metric in self.statistical_results[p][click_type] and
                            condition in self.statistical_results[p][click_type][metric]):

                        results = self.statistical_results[p][click_type][metric][condition]
                        test_values.append(results[test])
                    else:
                        test_values.append(np.nan)

                plt.plot([p * 100 for p in percentages], test_values,
                         marker='o',
                         label=condition,
                         color=get_condition_style(condition),
                         linewidth=2.5,
                         markersize=10)

            plt.axhline(y=0.05, color='r', linestyle=':', label='α = 0.05')
            plt.xlabel('Percentage of Simulated Data Added')
            plt.ylabel('p-value')
            plt.legend(title='Conditions',
                       bbox_to_anchor=(1.05, 1),
                       loc='upper left')
            # Set x-ticks at 10% intervals
            plt.xticks(np.arange(0, 101, 10))

            plt.yscale('log')

            # Remove legend from plot
            plt.legend().remove()

            plt.tight_layout()

            # Save in PDF format
            plt.savefig(
                self.results_augmented_dir / f'{metric}_{click_type}_{test}_results.pdf',
                format='pdf',
                bbox_inches='tight'
            )
            plt.close()

    def create_progression_plots(self):
        """Create visualization of metric progression"""
        print("Creating progression plots...")
        if not self.base_metrics or not self.progressive_metrics:
            raise ValueError("Must run both base and incremental analysis first")

        # Collect all metrics
        all_metrics = set()
        for condition in self.base_metrics:
            for click_type in ['choose', 'expand']:
                if click_type in self.base_metrics[condition]:
                    metrics = [m for m in self.base_metrics[condition][click_type].keys()
                               if 'time' not in m.lower()]
                    all_metrics.update(metrics)

        # Create plots
        for metric in sorted(all_metrics):
            for click_type in ['choose', 'expand']:
                self._create_single_metric_plot(click_type, metric)
                self._create_statistical_plot(click_type, metric)

    def _create_single_metric_plot(self, click_type: str, metric: str):
        """Create a single progression plot showing differences from base value"""
        ax = setup_plot_common()

        conditions = set(self.base_metrics.keys())

        # Sort conditions to ensure consistent ordering
        for condition in sorted(conditions):
            if metric not in self.base_metrics[condition][click_type]:
                continue

            base_value = np.mean(self.base_metrics[condition][click_type][metric])

            try:
                progression_values = []
                for p in sorted(self.progressive_metrics.keys()):
                    metrics, _ = self.progressive_metrics[p]
                    if (condition in metrics and click_type in metrics[condition] and
                            metric in metrics[condition][click_type]):
                        diff = metrics[condition][click_type][metric] - base_value
                        progression_values.append(diff)

                if progression_values:
                    percentages = [0] + [p * 100 for p in sorted(self.progressive_metrics.keys())]
                    values = [0] + progression_values  # Explicitly set first value to 0

                    plt.plot(percentages, values, marker='o',
                             label=condition,
                             color=get_condition_style(condition),
                             linewidth=2.5,
                             markersize=10)

            except Exception as e:
                print(f"Error plotting {condition}, {click_type}, {metric}: {e}")
                continue

        plt.xlabel('Percentage of Simulated Data Added')
        plt.ylabel(f'Difference in {metric.upper()} from Original')
        plt.legend(title='Conditions',
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left')
        # Set x-ticks at 10% intervals
        plt.xticks(np.arange(0, 101, 10))

        # Remove legend from plot
        plt.legend().remove()

        plt.tight_layout()

        # Save in PDF format
        plt.savefig(
            self.results_augmented_dir / f'{metric}_{click_type}_incremental_diff.pdf',
            format='pdf',
            bbox_inches='tight'
        )
        plt.close()

class MultiRunIncrementalAnalysis:
    """Wrapper class to handle multiple runs of the analysis"""

    def __init__(self, base_dir: str, n_runs: int = 5):
        self.n_runs = n_runs
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "multi_runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # Store results from each run
        self.run_results = []
        self.averaged_metrics = {}
        self.averaged_statistical_results = defaultdict(lambda: {
            'choose': defaultdict(dict),
            'expand': defaultdict(dict),
        })

    def _check_cached_results(self, run_number: int, percentages: List[float], y_values: List[int]) -> bool:
        """Check if cached results exist for a specific run"""
        run_dir = self.runs_dir / f"run_{run_number}"
        if not run_dir.exists():
            return False

        # Check for base metrics cache
        if not (run_dir / "results" / "base_metrics.pkl").exists():
            return False

        # Check for incremental caches
        cache_dir = run_dir / "cache"
        if not cache_dir.exists():
            return False

        # Check all required cache files exist
        for y in y_values:
            last_percentage = 0
            for current_percentage in percentages:
                cache_filename = f"process_logs_y{y}_pct{last_percentage:.3f}_{current_percentage:.3f}.pkl"
                if not (cache_dir / cache_filename).exists():
                    return False
                last_percentage = current_percentage

        return True

    def _load_cached_run(self, run_number: int, db_logs, percentages: List[float],
                         y_values: List[int]) -> IncrementalProgressiveAnalysis:
        """Load cached results for a specific run"""
        run_dir = self.runs_dir / f"run_{run_number}"

        # Create analysis instance
        run_analysis = IncrementalProgressiveAnalysis(
            simulation_dir=str(run_dir),
            results_dir=str(run_dir / "results"),
            results_augmented_dir=str(run_dir / "results_augmented")
        )

        # Load base metrics
        with open(run_dir / "results" / "base_metrics.pkl", 'rb') as f:
            run_analysis.base_metrics = pickle.load(f)

        # Load incremental results
        last_percentage = 0
        for current_percentage in percentages:
            all_metrics = []
            all_weights = []

            # Load cached results for each y value
            for y in y_values:
                cache_path = run_dir / "cache" / f"process_logs_y{y}_pct{last_percentage:.3f}_{current_percentage:.3f}.pkl"
                try:
                    with open(cache_path, 'rb') as f:
                        metrics = pickle.load(f)
                        if metrics and metrics[0]:
                            all_metrics.append(metrics[0])
                            # Calculate weight from the logs
                            weight = sum(len(logs.get_logs("confirmed")) for _, logs in db_logs.logs.items())
                            all_weights.append(weight)
                except Exception as e:
                    print(f"Error loading cache for y={y}, percentages={last_percentage}-{current_percentage}: {e}")
                    return None

            if all_metrics:
                # Combine metrics
                avg_metrics = run_analysis._weighted_average_metrics([run_analysis.base_metrics] + all_metrics,
                                                                     [run_analysis.base_weight] + all_weights)
                run_analysis.progressive_metrics[current_percentage] = (avg_metrics, sum(all_weights))

                # Combine raw metrics for statistical tests
                raw_combined = run_analysis.combine_raw_metrics([run_analysis.base_metrics] + all_metrics)
                run_analysis._run_statistical_tests(raw_combined, current_percentage)

            last_percentage = current_percentage

        return run_analysis


    def _shuffle_csv_files(self, simulation_dir: Path) -> Path:
        """Create a new directory with shuffled versions of CSV files"""
        run_dir = self.runs_dir / f"run_{len(self.run_results) + 1}"
        run_dir.mkdir(parents=True, exist_ok=True)

        for file in simulation_dir.glob('*.csv'):
            # Read CSV
            df = pd.read_csv(file)
            # Shuffle the dataframe
            df_shuffled = df.sample(frac=1).reset_index(drop=True)
            # Save shuffled version
            df_shuffled.to_csv(run_dir / file.name, index=False)

        return run_dir

    def _average_metrics(self):
        """Average metrics across all runs"""
        if not self.run_results:
            return

        # Initialize structures for averaging
        sum_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        count_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Sum up metrics across runs
        for run_metrics in self.run_results:
            for percentage, (metrics, _) in run_metrics.progressive_metrics.items():
                for condition in metrics:
                    for click_type in metrics[condition]:
                        for metric_name, value in metrics[condition][click_type].items():
                            if 'time' not in metric_name.lower():
                                sum_metrics[percentage][condition][f"{click_type}_{metric_name}"] += value
                                count_metrics[percentage][condition][f"{click_type}_{metric_name}"] += 1

        # Calculate averages
        for percentage in sum_metrics:
            self.averaged_metrics[percentage] = {}
            for condition in sum_metrics[percentage]:
                self.averaged_metrics[percentage][condition] = {}
                for metric_key, total in sum_metrics[percentage][condition].items():
                    count = count_metrics[percentage][condition][metric_key]
                    if count > 0:
                        self.averaged_metrics[percentage][condition][metric_key] = total / count

    def _average_statistical_results(self):
        """Average statistical results across all runs"""
        if not self.run_results:
            return

        # Initialize sum and count dictionaries for averaging
        sum_results = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))))
        count_results = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))))

        # Sum up results across runs
        for run in self.run_results:
            for percentage in run.statistical_results:
                for click_type in run.statistical_results[percentage]:
                    for metric in run.statistical_results[percentage][click_type]:
                        for condition in run.statistical_results[percentage][click_type][metric]:
                            for test_type, value in run.statistical_results[percentage][click_type][metric][
                                condition].items():
                                if not np.isnan(value):
                                    sum_results[percentage][click_type][metric][condition][test_type] += value
                                    count_results[percentage][click_type][metric][condition][test_type] += 1

        # Calculate averages
        for percentage in sum_results:
            for click_type in sum_results[percentage]:
                for metric in sum_results[percentage][click_type]:
                    for condition in sum_results[percentage][click_type][metric]:
                        self.averaged_statistical_results[percentage][click_type][metric][condition] = {}
                        for test_type in sum_results[percentage][click_type][metric][condition]:
                            count = count_results[percentage][click_type][metric][condition][test_type]
                            if count > 0:
                                avg_value = sum_results[percentage][click_type][metric][condition][test_type] / count
                                self.averaged_statistical_results[percentage][click_type][metric][condition][
                                    test_type] = avg_value

    def run_multiple_analyses(self, db_logs, percentages: List[float], y_values: List[int], force_reload=False):
        """Run multiple analyses with different shuffled datasets"""
        for run in range(self.n_runs):
            print(f"\n=== Starting Run {run + 1}/{self.n_runs} ===")

            if not force_reload and self._check_cached_results(run + 1, percentages, y_values):
                print(f"Loading cached results for run {run + 1}")
                run_analysis = self._load_cached_run(run + 1, db_logs, percentages, y_values)
                if run_analysis:
                    self.run_results.append(run_analysis)
                    continue

            # Create shuffled version of data
            shuffled_dir = self._shuffle_csv_files(Path("simulation"))

            # Create new analysis instance for this run
            run_analysis = IncrementalProgressiveAnalysis(
                simulation_dir=str(shuffled_dir),
                results_dir=str(shuffled_dir / "results"),
                results_augmented_dir=str(shuffled_dir / "results_augmented")
            )

            # Run analysis
            run_analysis.run_base_analysis(db_logs)
            run_analysis.run_incremental_analysis(percentages, y_values, force_reload)

            # Store results
            self.run_results.append(run_analysis)

        # Average results across runs
        self._average_metrics()
        self._average_statistical_results()

    def create_averaged_plots(self):
        """Create plots using averaged results"""
        print("\nCreating averaged plots...")

        # Collect all metrics and click types
        metrics = set()
        click_types = ['choose', 'expand']

        # Get first run to determine available metrics
        if not self.run_results:
            return

        first_run = self.run_results[0]
        for condition in first_run.base_metrics:
            for click_type in click_types:
                if click_type in first_run.base_metrics[condition]:
                    metrics.update([m for m in first_run.base_metrics[condition][click_type].keys()
                                    if 'time' not in m.lower()])

        # Create averaged plots
        for metric in sorted(metrics):
            for click_type in click_types:
                self._create_averaged_metric_plot(click_type, metric)
                self._create_averaged_statistical_plot(click_type, metric)

    def _create_averaged_metric_plot(self, click_type: str, metric: str):
        """Create a single progression plot showing averaged differences from base value"""
        ax = setup_plot_common()

        base_metrics = self.run_results[0].base_metrics
        conditions = set(base_metrics.keys())

        for condition in sorted(conditions):
            if metric not in base_metrics[condition][click_type]:
                continue

            base_value = np.mean(base_metrics[condition][click_type][metric])

            try:
                progression_values = []
                percentages = sorted(self.averaged_metrics.keys())

                for p in percentages:
                    metrics = self.averaged_metrics[p]
                    metric_key = f"{click_type}_{metric}"
                    if condition in metrics and metric_key in metrics[condition]:
                        diff = metrics[condition][metric_key] - base_value
                        progression_values.append(diff)

                if progression_values:
                    plot_percentages = [0] + [p * 100 for p in percentages]
                    plot_values = [0] + progression_values  # Explicitly set first value to 0

                    plt.plot(plot_percentages, plot_values,
                             marker='o',
                             label=condition,
                             color=get_condition_style(condition),
                             linewidth=2.5,
                             markersize=10)

            except Exception as e:
                print(f"Error plotting averaged {condition}, {click_type}, {metric}: {e}")
                continue

        plt.xlabel('Percentage of Simulated Data Added')
        plt.ylabel(f'Avg. Diff. in {metric.upper()}')
        plt.legend(title='Conditions',
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left')
        # Set x-ticks at 10% intervals
        plt.xticks(np.arange(0, 101, 10))

        # Remove legend from plot
        plt.legend().remove()

        plt.tight_layout()

        save_path = self.base_dir / "averaged_results" / f'{metric}_{click_type}_averaged_diff.pdf'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def _create_averaged_statistical_plot(self, click_type: str, metric: str):
        """Create separate plots showing averaged p-values from statistical tests"""
        tests = ['t_test', 'tost', 'welch_vs_base']
        test_labels = {
            't_test': 't-test',
            'tost': 'TOST',
            'welch_vs_base': "Welch's t-test vs Base"
        }

        percentages = sorted(self.averaged_statistical_results.keys())
        base_condition = "b=0.5,k1=0.8,batch_size=4"

        for test in tests:
            ax = setup_plot_common()

            conditions = set()
            for p in percentages:
                if click_type in self.averaged_statistical_results[p]:
                    if metric in self.averaged_statistical_results[p][click_type]:
                        conditions.update(self.averaged_statistical_results[p][click_type][metric].keys())

            for condition in sorted(conditions):
                if condition == base_condition:
                    continue

                test_values = []
                for p in percentages:
                    if (click_type in self.averaged_statistical_results[p] and
                            metric in self.averaged_statistical_results[p][click_type] and
                            condition in self.averaged_statistical_results[p][click_type][metric]):

                        value = self.averaged_statistical_results[p][click_type][metric][condition][test]
                        test_values.append(value)
                    else:
                        test_values.append(np.nan)

                plt.plot([p * 100 for p in percentages], test_values,
                         marker='o',
                         label=condition,
                         color=get_condition_style(condition),
                         linewidth=2.5,
                         markersize=10)

            plt.axhline(y=0.05, color='r', linestyle=':', label='α = 0.05')
            plt.xlabel('Percentage of Simulated Data Added')
            plt.ylabel('Average p-value')

            # Set x-ticks at 10% intervals
            plt.xticks(np.arange(0, 101, 10))
            plt.yscale('log')

            # Remove legend from plot
            plt.legend().remove()

            plt.tight_layout()

            save_path = self.base_dir / "averaged_results" / f'{metric}_{click_type}_{test}_averaged_results.pdf'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.close()


# Example usage
if __name__ == '__main__':
    SMALL_SIZE = 55
    MEDIUM_SIZE = 55
    BIGGER_SIZE = 55

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    try:
        # Create multi-run analysis instance
        multi_analysis = MultiRunIncrementalAnalysis(base_dir='simulation', n_runs=5)

        # Run base analysis
        db_logs = Logs("test_log")

        # Run incremental analysis with multiple runs
        percentages = [0.001, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        y_values = range(1, 6)

        # Run multiple analyses
        multi_analysis.run_multiple_analyses(db_logs, percentages, y_values)

        # Create averaged plots
        multi_analysis.create_averaged_plots()

        save_standalone_legend()

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()