"""Model Server Factory"""

from typing import List

from uniflow.model.config import ModelConfig


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
    def get(cls, name: str):
        """Get model server.

        Args:
            name (str): Model server name.

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

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize AbsModelServer class.

        Args:
            model_config (ModelConfig): Model config.
        """
        self._model_config = model_config


class OpenAIModelServer(AbsModelServer):
    """OpenAI Model Server Class."""

    def __init__(self, model_config: ModelConfig):
        # import in class level to avoid installing openai package
        from openai import OpenAI  # pylint: disable=import-outside-toplevel

        super().__init__(model_config)
        self._client = OpenAI()

    def _preprocess(self, data: str) -> str:
        """Preprocess data.

        Args:
            data (str): Data to preprocess.

        Returns:
            str: Preprocessed data.
        """
        return data

    def _postprocess(self, data: str) -> List[str]:
        return [c.message.content for c in data.choices]

    def __call__(self, data: str) -> str:
        """Run model.

        Args:
            data (str): Data to run.

        Returns:
            str: Output data.
        """
        data = self._preprocess(data)
        data = self._client.chat.completions.create(
            model=self._model_config.model_name,
            messages=[
                {"role": "user", "content": data},
            ],
            n=self._model_config.num_call,
            temperature=self._model_config.temperature,
            response_format=self._model_config.response_format,
        )
        data = self._postprocess(data)
        return data


class HuggingfaceModelServer(AbsModelServer):
    """Huggingface Model Server Class."""

    # Implement huggingface server here
