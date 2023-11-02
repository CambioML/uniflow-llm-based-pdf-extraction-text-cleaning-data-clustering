"""Output data operation."""
import copy

from typing import Any, Mapping
import os
import io
import json
import ast
import pandas as pd
import logging
from uniflow.op.basic.linear_op import LinearOp
from uniflow.op.utils import check_path_exists
from uniflow.flow.constants import (
    QAPAIR_DF_KEY,
    ERROR_LIST,
    OUTPUT_FILE,
    OUTPUT_AUGMENT_PREFIX,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataOutOp(LinearOp):
    """Output data operation.

    Args:
        nodes (Sequence[Node]): Input nodes.
    Returns:
        Sequence[Node]: Output nodes.
    """

    def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Output the augmented data to a json file.
            Then sanity check the augmented data and output it to a csv file.
        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict.
        """

        # --------- helper functions --------------------------------------
        def _make_w_io_base(f, mode: str):
            """Make a writable io base object from a string path to a file.
            Args:
                f: A string path to the location on disk.
                mode: Mode for opening the file.
            Returns:
                A writable io base object.
            """
            if not isinstance(f, io.IOBase):
                f_dirname = os.path.dirname(f)
                if f_dirname != "":
                    os.makedirs(f_dirname, exist_ok=True)
                f = open(f, mode=mode)
            return f

        def jdump(obj, f, mode="w", indent=4, default=str):
            """Dump a str or dictionary to a file in json format.

            Args:
                obj: An object to be written.
                f: A string path to the location on disk.
                mode: Mode for opening the file.
                indent: Indent for storing json dictionaries.
                default: A function to handle non-serializable entries; defaults to `str`.
            Returns:
                None
            """
            f = _make_w_io_base(f, mode)
            if isinstance(obj, (dict, list)):
                json.dump(obj, f, indent=indent, default=default)
            elif isinstance(obj, str):
                f.write(obj)
            else:
                raise ValueError(f"Unexpected type: {type(obj)}")
            f.close()

        # -----------------------------------------------------------------
        logger.info("Starting DataOutOp...")

        qaa_augmented_raw = copy.deepcopy(value_dict["qaa_augmented_raw"])
        filename = OUTPUT_AUGMENT_PREFIX + "data.json"
        current_directory = os.getcwd()
        output_dir = f"{current_directory}/data/output/"
        check_path_exists(output_dir)
        jdump(qaa_augmented_raw, os.path.join(output_dir, filename))

        QApair_dict = []
        error_list = []
        for raw in qaa_augmented_raw:
            pairs = raw.split("######")
            for _, item in enumerate(pairs):
                # print(id)
                if not item.isspace():
                    try:
                        str2dict = ast.literal_eval(item)
                        QApair_dict.append(str2dict)
                    except Exception as e:
                        print(e)
                        print(item)
                        print("\n\n\n")
                        error_list.append(item)
                        # print(type(str2dict))
                        # print(str2dict['_question'])
                        # print(str2dict['_answer'])
        QApair_df = pd.DataFrame(QApair_dict)
        filename = OUTPUT_AUGMENT_PREFIX + "data.csv"
        output_file = os.path.join(output_dir, filename)
        QApair_df.to_csv(output_file, index=False)
        logger.info("DataOutOp complete!")

        return {
            QAPAIR_DF_KEY: QApair_df,
            ERROR_LIST: error_list,
            OUTPUT_FILE: output_file,
        }
