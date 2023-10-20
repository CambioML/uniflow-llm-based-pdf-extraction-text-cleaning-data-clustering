"""prompt engineering operation."""
from typing import Any, Mapping
import re
from uniflow.op.basic.linear_op import LinearOp
import uniflow.flow.constants as constants


class PromptEngOp(LinearOp):
    """prompt engineering operation.

    Args:  nodes (Sequence[Node]): Input nodes.

    Returns: Sequence[Node]: Output nodes.
    """

    def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Provide a prompt that can be used to generate paraphrases of the
        input question and answer pairs.
            Encode prompt instruction and QA pairs into a single string.
        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict.
        """
        qaa = value_dict["qaa_processed"].copy(deep=True)
        qaa_list = qaa[[constants.QUESTION_KEY, constants.ANSWER_KEY]].to_dict("records")
        prompts = """Paraphrase the below question and answer pair in 3 different ways.
            Try not to repeat the verb for each pair to maximize diversity.
            Return everything in an array of JSON object in this format: ######{"_question":"string", "_answer":"string"}
            Seperate each pair with "######" rather than commas.
            """

        def encode_prompt_QA(prompts=prompts, QA_list=[]):
            """Encode multiple prompt instructions into a single string."""
            listof_prompt_QA = []
            for _, task_dict in enumerate(QA_list):
                single_prompt_QA = prompts + "\n"
                (question, answer) = task_dict[constants.QUESTION_KEY], task_dict[constants.ANSWER_KEY]
                question = re.sub(r"\s+", " ", question).strip().rstrip(":")
                answer = "<noinput>" if answer.lower() == "" else answer
                single_prompt_QA += f"######\n"
                single_prompt_QA += "{"
                single_prompt_QA += f'"_question": "{question}",\n'
                single_prompt_QA += f'"_answer": "{answer}"'
                single_prompt_QA += "}"
                # single_prompt_QA += f"###\n"
                listof_prompt_QA.append(single_prompt_QA)
            return listof_prompt_QA

        qaa_list_encoded = encode_prompt_QA(prompts, qaa_list)
        return {"qaa_list_encoded": qaa_list_encoded}
