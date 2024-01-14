"""LLM rater for tasks with a LLM model server."""

import re
from collections import Counter
from typing import Any, Dict, List

from uniflow.op.model.constants import (
    AVERAGE_SCORE,
    MAJORITY_VOTE,
    RESPONSE,
    SCORES,
    VOTES,
)
from uniflow.op.model.llm_processor import JsonFormattedDataProcessor, LLMDataProcessor
from uniflow.op.prompt import PromptTemplate


class LLMRater(LLMDataProcessor):
    """LLM Rater"""

    def __init__(
        self,
        prompt_template: PromptTemplate,
        model_config: Dict[str, Any],
        label2score: Dict[str, float],
    ) -> None:
        """LLM Rater Constructor.

        Args:
            prompt_template (PromptTemplate): Guided prompt template.
            model_config (Dict[str, Any]): Model config.
            label2score (Dict[str, float]): String to score mapping.
        """
        super().__init__(prompt_template, model_config)
        pattern = r"^[^A-Za-z]+|[^A-Za-z]+$"
        self._label2score = {
            re.sub(pattern, "", k).lower().lower(): float(v)
            for k, v in label2score.items()
        }
        self._score2label = {v: k for k, v in self._label2score.items()}

    def _deserialize(self, data: List[str]) -> List[Dict[str, Any]]:
        """Deserialize data.

        Args:
            data (List[str]): Data to deserialize.

        Returns:
            List[Dict[str, Any]]: Deserialized data.
        """

        def _extract_label(text):
            # Create a regular expression pattern from the dictionary keys
            # The pattern will look like '(yes|no)', case-insensitive
            pattern = (
                r"\b("
                + "|".join(  # noqa: W503
                    re.escape(key) for key in self._label2score.keys()
                )  # noqa: W503
                + r")\b"  # noqa: W503
            )

            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                # Return the last matched group (Yes or No), preserving the original case
                return (
                    matches[-1].group(1).lower()
                )  # Convert to lowercase to match dictionary keys
            else:
                # Return None if no match is found
                return None

        data = super()._deserialize(data)
        response = data[RESPONSE]

        labels = [_extract_label(d) for d in response]
        scores = []
        for label in labels:
            if label is not None:
                scores.append(self._label2score[label])
        majority_vote = Counter(labels).most_common(1)[0][0]
        mean_score = sum(scores) / len(scores) if len(scores) > 0 else None

        data.update(
            {
                MAJORITY_VOTE: majority_vote,
                AVERAGE_SCORE: mean_score,
                VOTES: labels,
                SCORES: scores,
            }
        )

        return data


class OpenAIJsonFormattedLLMRater(JsonFormattedDataProcessor):
    """OpenAI formatted LLM Rater"""

    def __init__(
        self,
        prompt_template: PromptTemplate,
        model_config: Dict[str, Any],
        label2score: Dict[str, float],
    ) -> None:
        """Json Formatted LLM Rater Constructor.

        Args:
            prompt_template (PromptTemplate): Guided prompt template.
            model_config (Dict[str, Any]): Model config.
            label2score (Dict[str, float]): String to score mapping.
        """
        super().__init__(prompt_template, model_config)
        self._pattern = r"^[^A-Za-z]+|[^A-Za-z]+$"
        self._label2score = {
            re.sub(self._pattern, "", k).lower(): float(v)
            for k, v in label2score.items()
        }
        self._score2label = {v: k for k, v in self._label2score.items()}
        self._rater_key = None
        if prompt_template.few_shot_prompt:
            example_keys = list(prompt_template.few_shot_prompt[0].dict().keys())
            self._rater_key = example_keys[-1]

    def _deserialize(self, data: List[str]) -> List[Dict[str, Any]]:
        """Deserialize data.

        Args:
            data (List[str]): Data to deserialize.

        Returns:
            List[Dict[str, Any]]: Deserialized data.
        """
        data = super()._deserialize(data)
        response = data[RESPONSE]
        if self._rater_key:
            labels = [
                re.sub(self._pattern, "", r[self._rater_key]).lower()
                if self._rater_key in r
                else None
                for r in response
            ]
        else:
            # If the rater key is not specified, use the last key in the response
            # as the rater key for the first response.
            self._rater_key = list(response[0].keys())[-1]
            labels = [
                re.sub(self._pattern, "", r[self._rater_key]).lower() for r in response
            ]
        scores = []
        for label in labels:
            if label is not None and label in self._label2score:
                scores.append(self._label2score[label])
        majority_vote = Counter(labels).most_common(1)[0][0]
        mean_score = sum(scores) / len(scores) if len(scores) > 0 else None
        data.update(
            {
                MAJORITY_VOTE: majority_vote,
                AVERAGE_SCORE: mean_score,
                VOTES: labels,
                SCORES: scores,
            }
        )

        return data


class HuggingfaceJsonFormattedLLMRater(LLMDataProcessor):
    """Huggingface formatted Json Formatted LLM Rater"""

    def __init__(
        self,
        prompt_template: PromptTemplate,
        model_config: Dict[str, Any],
        label2score: Dict[str, float],
    ) -> None:
        """Json Formatted LLM Rater Constructor.

        Args:
            prompt_template (PromptTemplate): Guided prompt template.
            model_config (Dict[str, Any]): Model config.
            label2score (Dict[str, float]): String to score mapping.
        """
        super().__init__(prompt_template, model_config)
        self._pattern = r"^[^A-Za-z]+|[^A-Za-z]+$"
        self._label2score = {
            re.sub(self._pattern, "", k).lower(): float(v)
            for k, v in label2score.items()
        }
        self._score2label = {v: k for k, v in self._label2score.items()}
        self._rater_key = None
        if prompt_template.few_shot_prompt:
            example_keys = list(prompt_template.few_shot_prompt[0].dict().keys())
            self._rater_key = example_keys[-1]

    def _deserialize(self, data: List[str]) -> List[Dict[str, Any]]:
        """Deserialize data.

        Args:
            data (List[str]): Data to deserialize.

        Returns:
            List[Dict[str, Any]]: Deserialized data.
        """
        data = super()._deserialize(data)
        response = data[RESPONSE]
        if self._rater_key:
            labels = [
                re.sub(self._pattern, "", r[self._rater_key]).lower()
                if self._rater_key in r
                else None
                for r in response
            ]
        else:
            # If the rater key is not specified, use the last key in the response
            # as the rater key for the first response.
            self._rater_key = list(response[0].keys())[-1]
            labels = [
                re.sub(self._pattern, "", r[self._rater_key]).lower() for r in response
            ]
        scores = []
        for label in labels:
            if label is not None and label in self._label2score:
                scores.append(self._label2score[label])
        majority_vote = Counter(labels).most_common(1)[0][0]
        mean_score = sum(scores) / len(scores) if len(scores) > 0 else None
        data.update(
            {
                MAJORITY_VOTE: majority_vote,
                AVERAGE_SCORE: mean_score,
                VOTES: labels,
                SCORES: scores,
            }
        )

        return data
