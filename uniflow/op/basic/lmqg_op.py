""" Operation that generates questions and answers from paragraphs."""
import subprocess
import copy
from typing import Any, Mapping
from uniflow.op.basic.linear_op import LinearOp
from lmqg import TransformersQG



class LMQGOp(LinearOp):
    """ Use LMQG to generate questions and answers from paragraphs.
    Args:
        nodes (Sequence[Node]): Input nodes.

    Returns:
        Sequence[Node]: Output nodes.
    """

    def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Call the language model by lmqg to generate outputs for the prompt.
        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict.
        """
        paragraphs = copy.deepcopy(value_dict["paragraphs"])



        # Download the en_core_web_sm model explicitly
        list_dir = subprocess.Popen(["python", "-m", "spacy", "download", "en_core_web_sm"])
        list_dir.wait()

         # Load the en_core_web_sm package in poetry
        #nlp = spacy.load("en_core_web_sm")

        # initialize model
        # set max length of a paragraph as 1024
        model = TransformersQG(model='lmqg/t5-base-squad-qg-ae', max_length=1024)
        # paragraph to generate pairs of question and answer

        question_answer = model.generate_qa(paragraphs)
        # the output is a list of tuple (question, answer)

        return {"qaa_raw": question_answer}
