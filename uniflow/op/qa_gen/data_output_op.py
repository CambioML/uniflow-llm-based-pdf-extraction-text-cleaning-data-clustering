"""Output data operation."""
import copy

from typing import Any, Mapping
import os
import io
import json
import ast
import pandas as pd
from uniflow.op.basic.linear_op import LinearOp



class DataOutOp(LinearOp):
    """Output data operation.

    Args:
        nodes (Sequence[Node]): Input nodes.
    Returns:
        Sequence[Node]: Output nodes.
    """

    def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """ Output the augmented data to a json file.
            Then sanity check the augmented data and output it to a csv file.
        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict.
        """
        # --------- helper functions --------------------------------------
        def _make_w_io_base(f, mode: str):
            """ Make a writable io base object from a string path to a file.
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

        qaa_augmented_raw = copy.deepcopy(value_dict["qaa_augmented_raw"])
        output_name = "output_qa_augment.json"
        current_directory = os.getcwd()
        output_dir = f"{current_directory}/tests/data/output/"
        jdump(qaa_augmented_raw, os.path.join(output_dir, output_name))

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
        output_name = "output_qa_augment.csv"
        QApair_df.to_csv(os.path.join(output_dir, output_name), index=False)

        return {"QApair_df": QApair_df, "error_list": error_list}
