"""Preprocess operation."""
from typing import Any, Mapping
import re
import logging
from uniflow.op.basic.linear_op import LinearOp
from cleantext import clean
import pandas as pd
from uniflow.flow.constants import QAPAIR_DF_KEY, QUESTION_KEY, ANSWER_KEY

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PreprocessOp(LinearOp):
    """Preprocess operation."""

    def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Transform value dict.

        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict.
        """

        logger.info("Starting PreprocessOp...")
        input_file = value_dict["input_file"]
        qaa = pd.read_csv(input_file, encoding="utf8")

        # Personal Identification Information (PII) removal and other preprocessing using cleantext

        def f_clean(qaa_pair_raw):
            return clean(
                qaa_pair_raw,
                fix_unicode=True,  # fix various unicode errors
                to_ascii=True,  # transliterate to closest ASCII representation
                lower=True,  # lowercase text
                # fully strip line breaks as opposed to only normalizing them
                no_line_breaks=False,
                no_urls=False,  # replace all URLs with a special token
                no_emails=False,  # replace all email addresses with a special token
                no_phone_numbers=False,  # replace all phone numbers with a special token
                no_numbers=False,  # replace all numbers with a special token
                no_digits=False,  # replace all digits with a special token
                no_currency_symbols=False,  # replace all currency symbols with a special token
                no_punct=False,  # remove punctuations
                replace_with_punct="",  # instead of removing punctuations you may replace them
                replace_with_url="<URL>",
                replace_with_email="<EMAIL>",
                replace_with_phone_number="<PHONE>",
                replace_with_number="<NUMBER>",
                replace_with_digit="0",
                replace_with_currency_symbol="<CUR>",
                lang="en",  # set to 'de' for German special handling
            )

        question_l_raw = qaa[QUESTION_KEY].to_list()
        question_l = [f_clean(p) for p in question_l_raw]
        answer_l_raw = qaa[ANSWER_KEY].to_list()
        answer_l = [f_clean(p) for p in answer_l_raw]
        qaa[QUESTION_KEY] = question_l
        qaa[ANSWER_KEY] = answer_l

        # Dataset customized clenaup
        answer_l_raw = qaa[ANSWER_KEY].to_list()
        qaa[ANSWER_KEY] = [
            re.compile(r"<.*?>|More\.\.\.", flags=re.IGNORECASE).sub("", p)
            for p in answer_l_raw
        ]  # Remove HTML tags/markups
        logger.info("PreprocessOp Complete!")

        return {"qaa_processed": qaa}
