"""Flow class."""
import os
from typing import Sequence
from dotenv import load_dotenv
from uniflow.node.node import Node
from uniflow.flow.flow import Flow
from uniflow.op.qa_gen.preprocess_op import PreprocessOp
from uniflow.op.qa_gen.prompt_eng_op import PromptEngOp
from uniflow.op.qa_gen.model_inf_op import ModelInfOp
from uniflow.op.qa_gen.data_output_op import DataOutOp
import openai

# NOTE: Configure your openai api key with python-dotenv.
# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")


class DataGenFlow(Flow):
    """Data generation flow class."""

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run flow.

        Args:
            nodes (Sequence[Node]): Nodes.

        Returns:
            Sequence[Node]: Nodes.
        """
        # Preprocessing for a list of datasets?
        proprocess_op = PreprocessOp("proprocess_op")
        node_preproc = proprocess_op(nodes)
        prompt_eng_op = PromptEngOp("prompt_eng_op")
        node_prompt_eng = prompt_eng_op(node_preproc)
        model_inf_op = ModelInfOp("model_inf_op")
        node_model_inf = model_inf_op(node_prompt_eng)
        data_out_op = DataOutOp("data_out_op")
        node_data_out = data_out_op(node_model_inf)
        return node_data_out
