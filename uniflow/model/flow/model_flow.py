"""Model Flow Module."""
from typing import Any, Dict, Sequence

from uniflow.flow import Flow
from uniflow.model.flow.flow_factory import FlowFactory
from uniflow.model.model import JsonModel, Model
from uniflow.node.node import Node
from uniflow.op.model.model_op import ModelOp
from uniflow.schema import GuidedPrompt


class ModelFlow(Flow):
    """Model Flow Class."""

    def __init_subclass__(cls) -> None:
        FlowFactory.register(name=cls.__name__, flow_cls=cls)
        super().__init_subclass__()


class BaseModelFlow(ModelFlow):
    """Model Base Flow Class."""

    def __init__(
        self,
        model_server: str,
        guided_prompt_template: GuidedPrompt,
        model_config: Dict[str, Any],
    ) -> None:
        """Model Flow Constructor.

        Args:
            model_server (str): Model server name.
            guided_prompt_template (GuidedPrompt): Guided prompt template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__()
        self._model_op = ModelOp(
            name="model_op",
            model=Model(
                model_server=model_server,
                guided_prompt_template=guided_prompt_template,
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


class OpenAIModelFlow(ModelFlow):
    """OpenAI Model Flow Class."""

    def __init__(
        self,
        model_server: str,
        guided_prompt_template: GuidedPrompt,
        model_config: Dict[str, Any],
    ) -> None:
        """OpenAI Model Flow Constructor.

        Args:
            model_server (str): Model server name.
            guided_prompt_template (GuidedPrompt): Guided prompt template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__()
        if model_config["response_format"]["type"] == "json_object":
            model = JsonModel(
                model_server=model_server,
                guided_prompt_template=guided_prompt_template,
                model_config=model_config,
            )
        else:
            model = Model(
                model_server=model_server,
                guided_prompt_template=guided_prompt_template,
                model_config=model_config,
            )
        self._model_op = ModelOp(
            name="openai_model_op",
            model=model,
        )

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        return self._model_op(nodes)


class HuggingFaceModelFlow(ModelFlow):
    """HuggingFace Model Flow Class."""

    def __init__(
        self,
        model_server: str,
        guided_prompt_template: GuidedPrompt,
        model_config: Dict[str, Any],
    ) -> None:
        """HuggingFace Model Flow Constructor.

        Args:
            model_server (str): Model server name.
            guided_prompt_template (GuidedPrompt): Guided prompt template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__()
        self._model_op = ModelOp(
            name="huggingface_model_op",
            model=Model(
                model_server=model_server,
                guided_prompt_template=guided_prompt_template,
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


class LMQGModelFlow(ModelFlow):
    """LMQG Model Flow Class."""

    def __init__(
        self,
        model_server: str,
        guided_prompt_template: GuidedPrompt,
        model_config: Dict[str, Any],
    ) -> None:
        """HuggingFace Model Flow Constructor.

        Args:
            model_server (str): Model server name.
            guided_prompt_template (GuidedPrompt): Guided prompt template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__()
        self._model_op = ModelOp(
            name="lmqg_model_op",
            model=Model(
                model_server=model_server,
                guided_prompt_template=guided_prompt_template,
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
