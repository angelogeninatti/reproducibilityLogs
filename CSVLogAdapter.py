import pandas as pd
from datetime import datetime
import json
from collections import defaultdict

from logs import Logs, Timeline


class CSVLogAdapter:
    def __init__(self, data):
        """
        Initialize with either a file path or a pandas DataFrame
        Args:
            data: Either a string (file path) or pandas DataFrame
        """
        if isinstance(data, pd.DataFrame):
            self.df = data
        else:
            self.df = pd.read_csv(data)
        self.logs = self._transform_csv_to_logs()

    def _get_query_info(self, row, clicked_index=None):
        """Helper method to create query info dict"""
        query_info = {
            'query_id': row['query_id'],
            'b': row['condition_b'],
            'k1': row['condition_k1'],
            'batch_size': 4
        }

        # If clicked_index is provided, add document information
        if clicked_index is not None:
            results_list = row['query_results'].split(',')
            if 0 <= clicked_index < len(results_list):
                query_info['docno'] = results_list[clicked_index]

        return query_info

    def _create_search_results_log(self, row):
        """Create a search_results log entry from a CSV row"""
        # Split query_results into a list of document IDs
        results = row['query_results'].split(',')

        # Create results list with docno for each result
        results_list = [{'docno': docid, 'rank': idx} for idx, docid in enumerate(results)]

        # Create query info including the first document's docno
        query_info = self._get_query_info(row)
        if results:  # If we have any results
            query_info['docno'] = results[0]

        return {
            'log': 'search_results',
            'exp_condition': {
                'b': row['condition_b'],
                'k1': row['condition_k1'],
                'batch_size': 4
            },
            'query': query_info,
            'results': results_list,
            'exp_step': 0
        }

    def _create_expand_result_log(self, row):
        """Create an expand_result log entry from a CSV row"""
        clicked_index = row['clicked_document_index']
        results_list = row['query_results'].split(',')

        return {
            'log': 'expand_result',
            'query': self._get_query_info(row, clicked_index),
            'index': clicked_index,
            'docno': results_list[clicked_index] if 0 <= clicked_index < len(results_list) else None
        }

    def _create_choose_result_log(self, row):
        """Create a choose_result and confirmed log entries from a CSV row"""
        clicked_index = row['clicked_document_index']
        results_list = row['query_results'].split(',')

        # Create choose_result log
        choose_log = {
            'log': 'choose_result',
            'query': self._get_query_info(row, clicked_index),
            'index': clicked_index
        }

        # Create chosen_results list with full document info
        chosen_result = {
            'b': row['condition_b'],
            'k1': row['condition_k1'],
            'batch_size': 4,
            'rank': clicked_index,
            'query_id': row['query_id'],
            'docno': results_list[clicked_index] if 0 <= clicked_index < len(results_list) else None
        }

        # Create confirmed log
        confirmed_log = {
            'log': 'confirmed',
            'query': self._get_query_info(row, clicked_index),
            'chosen_indices': [clicked_index],
            'chosen_results': [chosen_result],
        }

        return choose_log, confirmed_log

    def _transform_csv_to_logs(self):
        """Transform CSV data into the expected log format"""
        logs = {}

        # TODO complete here: append to user id b and k1.
        self.df['user_id'] = self.df.apply(
            lambda row: f"{row['user_id']}_{row['condition_b']}_{row['condition_k1']}", axis=1)
        # Group by user_id
        grouped = self.df.groupby('user_id')
        for user_id, user_data in grouped:
            timeline = Timeline(user_id)
            timeline.edit()

            # Process each interaction for the user
            for _, row in user_data.iterrows():
                # Create a base timestamp (can be adjusted if timestamps are available)
                timestamp = datetime.now()

                # Add search_results event
                search_results_log = self._create_search_results_log(row)
                timeline.add_event('search_results', search_results_log, timestamp)

                # Add expand_result event if there was a click
                if not pd.isna(row['clicked_document_index']):
                    expand_log = self._create_expand_result_log(row)
                    timeline.add_event('expand_result', expand_log, timestamp)

                    # Add choose_result and confirmed events
                    choose_log, confirmed_log = self._create_choose_result_log(row)
                    timeline.add_event('choose_result', choose_log, timestamp)
                    timeline.add_event('confirmed', confirmed_log, timestamp)

            # Add user_created event with experimental condition
            first_row = user_data.iloc[0]
            user_created_log = {
                'log': 'user_created',
                'exp_condition': {
                    'b': first_row['condition_b'],
                    'k1': first_row['condition_k1'],
                    'batch_size': 4
                }
            }
            timeline.add_event('user_created', user_created_log, timestamp)

            # Add end_experiment event
            end_experiment_log = {
                'log': 'end_experiment'
            }
            timeline.add_event('end_experiment', end_experiment_log, timestamp)

            timeline.close()
            logs[user_id] = timeline

        return logs

    def get_logs(self):
        """Return the transformed logs in the expected format"""
        return self.logs

def create_combined_logs(db_logs, csv_logs):
    """Combine logs from database and CSV sources"""
    # Create a new Logs instance with combined data
    combined_logs = Logs("test_log")  # Create empty logs object

    # Since csv_logs is already a dictionary of Timeline objects,
    # we can directly merge it with db_logs
    combined_logs.logs = {**db_logs.logs, **csv_logs}  # Merge both dictionaries

    return combined_logs