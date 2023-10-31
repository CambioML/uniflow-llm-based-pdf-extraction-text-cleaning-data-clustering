"""Output data generated from descriptive text operation."""
import copy

from typing import Any, Mapping
import os
import pandas as pd
from uniflow.op.basic.linear_op import LinearOp
from uniflow.flow.constants import QUESTION_KEY, ANSWER_KEY, ERROR_LIST, QAPAIR_DF_KEY
from uniflow.op.utils import check_path_exists


class DataOutTextOp(LinearOp):
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

        qaa_raw = copy.deepcopy(value_dict["qaa_raw"])
        error_list = copy.deepcopy(value_dict[ERROR_LIST])
        QApair_flat = [qa for qas in qaa_raw for qa in qas]

        QApair_df = pd.DataFrame(QApair_flat, columns=[QUESTION_KEY, ANSWER_KEY])
        dir_cur = os.getcwd()
        save_path = os.path.join(dir_cur, "data/output")
        check_path_exists(save_path)
        QApair_df.to_csv(os.path.join(save_path, "output_qa_text.csv"), index=False)

        return {QAPAIR_DF_KEY: QApair_df, ERROR_LIST: error_list}
