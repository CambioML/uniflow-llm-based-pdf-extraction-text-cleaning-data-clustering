"""
All Model Servers including ModelServerFactory, AbsModelServer.
"""

import logging
from typing import Any, Dict, List

from uniflow.op.prompt import PromptTemplate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
###############################################################################
#                             All Model Servers                               #
###############################################################################


class ModelServerFactory:
    """Model Server Factory."""

    _servers = {}

    @classmethod
    def register(cls, name: str, server_cls: "AbsModelServer") -> None:
        """Register model server.

        Args:
            name (str): Model server name.
            server_cls (AbsModelServer): Model server class.
        """
        cls._servers[name] = server_cls

    @classmethod
    def get(cls, name: str) -> "AbsModelServer":
        """Get model server.

        Args:
            name (str): Model server name.

        Returns:
            AbsModelServer: Model server.

        Raises:
            ValueError: If no model server registered under the name.
        """
        server_cls = cls._servers.get(name)
        if not server_cls:
            raise ValueError(f"No model server registered under '{name}'")
        return server_cls

    @classmethod
    def list(cls):
        """List all registered model servers.

        Returns:
            List[str]: List of registered model server names.
        """
        return list(cls._servers.keys())


class AbsModelServer:
    """Abstract Model Server Class."""

    def __init_subclass__(cls):
        """Register model server.

        This method is called when subclass is created.
        """
        super().__init_subclass__()
        ModelServerFactory.register(cls.__name__, cls)

    def __init__(
        self,
        prompt_template: PromptTemplate,
        model_config: Dict[str, Any],
    ) -> None:
        """Initialize AbsModelServer class.

        Args:
            prompt_template (PromptTemplate): Prompt template.
            model_config (Dict[str, Any]): Model config.
        """
        self._model_config = model_config
        self._example_keys = None
        if (
            prompt_template is not None
            and "few_shot_prompt" in prompt_template.model_fields  # noqa: W503
            and prompt_template.few_shot_prompt  # noqa: W503
        ):
            self._example_keys = list(
                prompt_template.few_shot_prompt[0].model_dump().keys()
            )

    def _preprocess(self, data: str) -> str:
        """Preprocess data.

        Args:
            data (str): Data to preprocess.

        Returns:
            str: Preprocessed data.
        """
        raise NotImplementedError

    def __call__(self, data: str) -> str:
        """Run model.

        Args:
            data (str): Data to run.

        Returns:
            str: Output data.
        """
        raise NotImplementedError

    def _postprocess(self, data: str) -> List[str]:
        """Postprocess data.

        Args:
            data (str): Data to postprocess.

        Returns:
            List[str]: Postprocessed data.
        """
        raise NotImplementedError
