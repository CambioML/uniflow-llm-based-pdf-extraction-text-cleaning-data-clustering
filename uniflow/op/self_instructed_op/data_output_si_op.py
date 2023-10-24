"""Output data generated from descriptive text operation."""
import copy

from typing import Any, Mapping
import os
import logging
import pandas as pd
from uniflow.op.basic.linear_op import LinearOp
from uniflow.flow.constants import ANSWER_KEY, QAPAIR_DF_KEY, QUESTION_KEY
from uniflow.op.utils import check_path_exists


class DataOutSIOp(LinearOp):
    """Output data operation.

    Args:
        nodes (Sequence[Node]): Input nodes.
    Returns:
        Sequence[Node]: Output nodes.
    """

    def __init__(self, name: str):
        """Initialize DataOutSIOp class."""
        super().__init__(name)
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

    def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Write generated question answer pairs to csv file
        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict.
        """
        # -----------------------------------------------------------------
        self._logger.info("Starting DataOutSIOp...")
        QApair_df = pd.DataFrame()

        QApair_df[QUESTION_KEY] = value_dict["text_line_q"]
        QApair_df["input"] = value_dict["text_line_in"]
        QApair_df[ANSWER_KEY] = value_dict["text_line_a"]

        dir_cur = os.getcwd()
        save_path = os.path.join(dir_cur, "data/output")
        check_path_exists(save_path)
        QApair_df.to_csv(
            os.path.join(save_path, "QA_output_self-instruct.csv"),
            index=False,
        )

        self._logger.debug(f"Created {QAPAIR_DF_KEY}: {QApair_df}")
        self._logger.info("DataOutSIOp complete!")

        return {QAPAIR_DF_KEY: QApair_df}
