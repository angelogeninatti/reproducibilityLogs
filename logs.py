import json
import os
import pickle
import warnings
from bidict import bidict
from functools import lru_cache
from collections import defaultdict
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from pprint import pprint
import numpy as np
import pandas as pd

from dataframe_getter import getDataframe


class Questions:
    def __init__(self):
        self.tests = dict()  # Test_name => list of questions
        self._question_indices = defaultdict(dict)  # Test_name => {question => index}

    def add_question(self, test_name: str, question: str) -> int:
        if test_name not in self.tests:
            self.tests[test_name] = []

        # Use cached index if available
        if question in self._question_indices[test_name]:
            return self._question_indices[test_name][question]

        # Add new question and cache its index
        if question not in self.tests[test_name]:
            self.tests[test_name].append(question)
        index = self.tests[test_name].index(question)
        self._question_indices[test_name][question] = index
        return index

    def __getitem__(self, test_name: str) -> List[str]:
        return self.tests[test_name]


questions = Questions()


class Event:
    def __init__(self, event_name: str, timestamp: datetime):
        self.event_name = event_name
        self.timestamp = timestamp

    def __str__(self) -> str:
        return '[Generic event] ' + self.event_name


class LogEvent(Event):
    def __init__(self, log_name: str, log_data: Dict, timestamp: datetime):
        super().__init__(log_name, timestamp)
        self.log_information = log_data
        self._cache = {}  # Cache for computed values

    def __getitem__(self, item: Any) -> Any:
        if item in self._cache:
            return self._cache[item]

        result = None
        if isinstance(self.log_information, list):
            result = self.log_information[item]
        else:
            result = self.log_information.get(item)

        self._cache[item] = result
        return result

    def __str__(self) -> str:
        return f'[Log] {self.event_name} at {self.timestamp} - {str(self.log_information)}\n'


class AnswersEvent(Event):
    def __init__(self, test_name: str, timestamp: datetime):
        super().__init__(test_name, timestamp)
        self.answers = dict()  # question_index => answer
        self._question_lookup = {}  # Cache for question lookups

    def add_answer(self, question: str, answer: Any) -> None:
        index = questions.add_question(self.event_name, question)
        self.answers[index] = answer
        self._question_lookup[question] = index

    def __getitem__(self, item: str) -> Any:
        if item in self._question_lookup:
            index = self._question_lookup[item]
        else:
            index = questions[self.event_name].index(item)
            self._question_lookup[item] = index
        return self.answers[index]

    def __str__(self) -> str:
        tmpstr = ""
        for index, answer in self.answers.items():
            tmpstr += f"{questions[self.event_name][index]} - {str(answer)}\n"
        return f'[Answers] {self.event_name}\n{tmpstr}'


class Timeline:
    def __init__(self, user: str):
        self.user = user
        self.events: List[Event] = []
        self.test_names: Dict[str, int] = {}  # Test_name => index in events
        self.editing = True

        # Indexes for faster lookups
        self._log_event_index: Dict[str, List[int]] = defaultdict(list)  # log_name => list of indices
        self._answer_event_index: Dict[str, int] = {}  # test_name => index
        self._event_type_index: Dict[type, List[int]] = defaultdict(list)  # event_type => list of indices

        # Cache for expensive operations
        self._cached_logs = {}
        self._cached_answers = {}

    def _rebuild_indices(self) -> None:
        self._log_event_index.clear()
        self._answer_event_index.clear()
        self._event_type_index.clear()

        for idx, event in enumerate(self.events):
            if isinstance(event, LogEvent):
                self._log_event_index[event.event_name].append(idx)
                self._event_type_index[LogEvent].append(idx)
            elif isinstance(event, AnswersEvent):
                self._answer_event_index[event.event_name] = idx
                self._event_type_index[AnswersEvent].append(idx)

    def edit(self) -> None:
        self.editing = True
        # Clear caches when editing starts
        self._cached_logs.clear()
        self._cached_answers.clear()

    def close(self) -> None:
        self.events.sort(key=lambda x: x.timestamp)
        self._rebuild_indices()
        self.editing = False

    def check_access(self) -> None:
        if self.editing:
            raise RuntimeError('In order to access a Timeline, you need to close it first: timeline.close().')

    def check_edit(self) -> None:
        if not self.editing:
            raise RuntimeError('In order to edit a Timeline, you need to call timeline.edit() first.')

    def add_event(self, test_name: str, log_data: Dict, timestamp: datetime) -> None:
        self.check_edit()

        if test_name == "answered_questions": # TODO
            test_name_full = f"{test_name}_{log_data['exp_step']}"
            if test_name_full not in self.test_names:
                event = AnswersEvent(test_name_full, timestamp)
                self.events.append(event)
                self.test_names[test_name_full] = len(self.events) - 1

                for question in log_data['questions']:
                    if "Please select " not in question['text']:
                        event.add_answer(question['text'], question['answer'])
        else:
            self.events.append(LogEvent(log_data.get('log'), log_data, timestamp))

    @lru_cache(maxsize=128)
    def get_logs(self, log_name: Optional[str] = None, precise: bool = True) -> List[LogEvent]:
        self.check_access()

        cache_key = (log_name, precise)
        if cache_key in self._cached_logs:
            return self._cached_logs[cache_key]

        if log_name is None:
            result = [self.events[idx] for idx in self._event_type_index[LogEvent]]
        else:
            indices = []
            if precise:
                indices = self._log_event_index.get(log_name, [])
            else:
                for name, idx_list in self._log_event_index.items():
                    if log_name in name:
                        indices.extend(idx_list)
            result = [self.events[idx] for idx in indices]

        self._cached_logs[cache_key] = result
        return result

    def get_logs_by_custom_function(self, fct: Callable[[LogEvent], bool]) -> List[LogEvent]:
        self.check_access()
        return [event for event in self.get_logs() if fct(event)]

    def get_events(self, event_name: Optional[str] = None, precise: bool = True) -> List[Event]:
        self.check_access()
        if event_name is None:
            return self.events

        return [event for event in self.events if
                (precise and event.event_name == event_name) or
                (not precise and event_name in event.event_name)]

    def get_answers(self, test_name: Optional[str] = None) -> List[AnswersEvent]:
        self.check_access()

        if test_name in self._cached_answers:
            return self._cached_answers[test_name]

        if test_name is None:
            result = [self.events[idx] for idx in self._event_type_index[AnswersEvent]]
        else:
            idx = self._answer_event_index.get(test_name)
            result = [self.events[idx]] if idx is not None else []

        self._cached_answers[test_name] = result
        return result

    def get_latest_log_before_event(self, log_name: str, event: Event, precise: bool = False) -> LogEvent:
        self.check_access()
        event_index = self.events.index(event)

        relevant_logs = self.get_logs(log_name, precise)
        for log in reversed(relevant_logs):
            if self.events.index(log) < event_index:
                return log

        raise FileNotFoundError(f'No {log_name} log was found prior to the event {str(event)}')

    def get_next_log_after_event(self, log_name: str, event: Event, precise: bool = False) -> LogEvent:
        self.check_access()
        event_index = self.events.index(event)

        relevant_logs = self.get_logs(log_name, precise)
        for log in relevant_logs:
            if self.events.index(log) > event_index:
                return log

        raise FileNotFoundError(f'No {log_name} log was found after the event {str(event)}')

    def get_events_between(self, start_event: Event, end_event: Event,
                           event_name: Optional[str] = None) -> List[Event]:
        self.check_access()
        start_idx = self.events.index(start_event) + 1
        end_idx = self.events.index(end_event)

        events = self.events[start_idx:end_idx]
        if event_name is None:
            return events
        return [event for event in events if event.event_name == event_name]

    @lru_cache(maxsize=128)
    def get_partition(self, partition_name: str, threshold: float,
                      system: Optional[str] = None) -> str:
        self.check_access()
        data = self.get_answers(partition_name)[0]
        if system is not None:
            data = data[system]
        else:
            data = data.answers[0]
        return f'high_{partition_name}' if data >= threshold else f'low_{partition_name}'

    def __str__(self) -> str:
        self.check_access()
        return f"TIMELINE FOR USER {self.user}\n" + \
            "".join(f"{str(event)}----------\n" for event in self.events)


class Logs:
    def __init__(self, table: Optional[str] = None,
                 preprocessed_data: Optional[Dict[str, Timeline]] = None,
                 dataframe: Optional[pd.DataFrame] = None,
                 force_reload: bool = False):
        self.logs: Dict[str, Timeline] = {}
        self._cache = {}

        if preprocessed_data is not None:
            self.logs = preprocessed_data
        else:
            if dataframe is None:
                results = getDataframe(table, force_reload)
            else:
                results = dataframe

            for _, row in results.iterrows():
                if row['user_id'] not in self.logs:
                    self.logs[row['user_id']] = Timeline(row['user_id'])

                log_data = json.loads(row['log_data'])
                self.logs[row['user_id']].add_event(
                    log_data['log'], log_data, row['timestamp'])

        for timeline in self.logs.values():
            timeline.close()

    def items(self):
        return self.logs.items()

    def __str__(self) -> str:
        return "".join(f"LOGS FOR USER {user}\n{timeline}\n\n"
                       for user, timeline in self.logs.items())