""" Operation that generates questions and answers from paragraphs."""
import subprocess
import copy
from typing import Any, Mapping
from uniflow.op.basic.linear_op import LinearOp
from lmqg import TransformersQG
from uniflow.flow.constants import ERROR_LIST
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LMQGOp(LinearOp):
    """Use LMQG to generate questions and answers from paragraphs.
    Args:
        nodes (Sequence[Node]): Input nodes.

    Returns:
        Sequence[Node]: Output nodes.
    """

    def __init__(self, name: str):
        """Initialize LMQGOp class."""
        super().__init__(name)

        logger.info("Initializing LMQGOp...")
        # initialize model
        # set max length of a paragraph as 1024
        self._model = TransformersQG(model="lmqg/t5-base-squad-qg-ae", max_length=1024)
        logger.info("LMQGOp initialization complete!")

    def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Call the language model by lmqg to generate outputs for the prompt.
        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict.
        """
        paragraphs = copy.deepcopy(value_dict["paragraphs"])

        # Download the en_core_web_sm model explicitly
        # list_dir = subprocess.Popen(
        #     ["python", "-m", "spacy", "download", "en_core_web_sm"]
        # )
        # list_dir.wait()

        # Load the en_core_web_sm package in poetry
        # nlp = spacy.load("en_core_web_sm")

        question_answer = []
        error_list = []

        for i, paragraph in enumerate(paragraphs):
            logger.info(
                f"Generating question and answer pairs for paragraph {i + 1} of {len(paragraphs)}"
            )
            try:
                output = self._model.generate_qa([paragraph])
                question_answer.extend(output)
                # the output is a list of tuple (question, answer)
            except Exception as e:
                logger.warning(f"Exception in paragraph {i + 1}: {repr(e)}")
                error_list.append({"paragraph": paragraph, "error": repr(e)})

        return {"qaa_raw": question_answer, ERROR_LIST: error_list}
