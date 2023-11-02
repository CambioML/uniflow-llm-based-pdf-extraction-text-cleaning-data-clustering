"""Output data generated from descriptive text operation."""
import copy

from typing import Any, Mapping
import os
import logging
import pandas as pd
from uniflow.op.basic.linear_op import LinearOp
from uniflow.flow.constants import (
    ANSWER_KEY,
    QAPAIR_DF_KEY,
    QUESTION_KEY,
    OUTPUT_FILE,
    OUTPUT_SELF_INSTRUCT_PREFIX,
)
from uniflow.op.utils import check_path_exists

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataOutSIOp(LinearOp):
    """Output data operation.

    Args:
        nodes (Sequence[Node]): Input nodes.
    Returns:
        Sequence[Node]: Output nodes.
    """

    def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Write generated question answer pairs to csv file
        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict.
        """
        # -----------------------------------------------------------------
        logger.info("Starting DataOutSIOp...")
        QApair_df = pd.DataFrame()

        QApair_df[QUESTION_KEY] = value_dict["text_line_q"]
        QApair_df[ANSWER_KEY] = value_dict["text_line_a"]

        dir_cur = os.getcwd()
        save_path = os.path.join(dir_cur, "data/output")
        check_path_exists(save_path)
        filename = OUTPUT_SELF_INSTRUCT_PREFIX + "data.csv"
        output_file = os.path.join(save_path, filename)
        QApair_df.to_csv(
            output_file,
            index=False,
        )

        logger.debug(f"Created {QAPAIR_DF_KEY}: {QApair_df}")
        logger.info("DataOutSIOp complete!")

        return {QAPAIR_DF_KEY: QApair_df, OUTPUT_FILE: output_file}
