"""Output data generated from descriptive text operation."""
import copy

from typing import Any, Mapping
import os
import pandas as pd
from uniflow.op.basic.linear_op import LinearOp
import uniflow.flow.constants as constants



class DataOutSIOp(LinearOp):
    """Output data operation.

    Args:
        nodes (Sequence[Node]): Input nodes.
    Returns:
        Sequence[Node]: Output nodes.
    """

    def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """ Write generated question answer pairs to csv file
        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict.
        """
        # -----------------------------------------------------------------

        print("Starting DataOutSIOp!")
        QApair_df = pd.DataFrame()
        print("text_line_in", value_dict["text_line_in"])

        QApair_df[constants.QUESTION_KEY] = value_dict["text_line_q"]
        QApair_df["input"] = value_dict["text_line_in"]
        QApair_df[constants.ANSWER_KEY] = value_dict["text_line_a"]

        # qaa_raw = copy.deepcopy(value_dict["qaa_raw"])
        # QApair_flat = [qa for qas in qaa_raw for qa in qas]

        dir_cur = os.getcwd()
        QApair_df.to_csv(os.path.join(f"{dir_cur}/data/output", "QA_output_selfinstruct.csv"), index=False)


        # QApair_df = pd.DataFrame(QApair_flat, columns=[constants.QUESTION_KEY, constants.ANSWER_KEY])
        # dir_cur = os.getcwd()
        # QApair_df.to_csv(os.path.join(f"{dir_cur}/data/output", "output_qa_text.csv"), index=False)

        print(f"Created {constants.QAPAIR_DF_KEY}: {QApair_df}")
        print("DataOutSIOp complete!")

        return {constants.QAPAIR_DF_KEY: QApair_df}
