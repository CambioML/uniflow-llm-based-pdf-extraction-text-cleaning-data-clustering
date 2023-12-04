"""Model Flow Module."""
from typing import Any, Dict, Sequence

from uniflow.flow.flow import Flow
from uniflow.model.model import JsonModel, Model, OpenAIJsonModel
from uniflow.node.node import Node
from uniflow.op.model.model_op import ModelOp


class ModelFlow(Flow):
    """Model Flow Class."""

    def __init__(
        self,
        model_server: str,
        few_shot_template: Dict[str, Any],
        model_config: Dict[str, Any],
    ) -> None:
        """Model Flow Constructor.

        Args:
            model_server (str): Model server name.
            few_shot_template (Dict[str, Any]): Few shot template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__()
        self._model_op = ModelOp(
            name="model_op",
            model=JsonModel(
                model_server=model_server,
                few_shot_template=few_shot_template,
                model_config=model_config,
            ),
        )

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        return self._model_op(nodes)


class OpenAIJsonModelFlow(Flow):
    """OpenAI Json Model Flow Class."""

    def __init__(
        self,
        model_server: str,
        few_shot_template: Dict[str, Any],
        model_config: Dict[str, Any],
    ) -> None:
        """OpenAI Json Model Flow Constructor.

        Args:
            model_server (str): Model server name.
            few_shot_template (Dict[str, Any]): Few shot template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__()
        self._model_op = ModelOp(
            name="openai_json_model_op",
            model=OpenAIJsonModel(
                model_server=model_server,
                few_shot_template=few_shot_template,
                model_config=model_config,
            ),
        )

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        return self._model_op(nodes)


class HuggingFaceModelFlow(Flow):
    """HuggingFace Model Flow Class."""

    def __init__(
        self,
        model_server: str,
        few_shot_template: Dict[str, Any],
        model_config: Dict[str, Any],
    ) -> None:
        """HuggingFace Model Flow Constructor.

        Args:
            model_server (str): Model server name.
            few_shot_template (Dict[str, Any]): Few shot template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__()
        self._model_op = ModelOp(
            name="huggingface_model_op",
            model=Model(
                model_server=model_server,
                few_shot_template=few_shot_template,
                model_config=model_config,
            ),
        )

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        return self._model_op(nodes)


class LMQGModelFlow(Flow):
    """LMQG Model Flow Class."""

    def __init__(
        self,
        model_server: str,
        few_shot_template: Dict[str, Any],
        model_config: Dict[str, Any],
    ) -> None:
        """HuggingFace Model Flow Constructor.

        Args:
            model_server (str): Model server name.
            few_shot_template (Dict[str, Any]): Few shot template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__()
        self._model_op = ModelOp(
            name="lmqg_model_op",
            model=Model(
                model_server=model_server,
                few_shot_template=few_shot_template,
                model_config=model_config,
            ),
        )

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        return self._model_op(nodes)
