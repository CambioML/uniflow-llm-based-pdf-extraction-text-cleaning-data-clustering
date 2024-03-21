import json
import os
from typing import Any, Dict


class BedrockEmbeddingClient:
    """Client for embedding service."""

    def __init__(self, aws_session, loader_config: Dict[str, Any]) -> None:
        """
        Initializes an instance of the Embedding class.

        Args:
            loader_config (Dict[str, Any]): A dictionary containing the loader configuration.

        Raises:
            ModuleNotFoundError: If the 'boto3' Python package is not installed.
            ValueError: If there is an issue with loading the AWS credentials.

        """

        self._aws_region = (
            loader_config["aws_region"] if "aws_region" in loader_config else None
        )

        # model_id: str = "amazon.titan-embed-text-v1"
        # model_kwargs: Optional[Dict[str, Any]] = {}
        self.model_id = loader_config["model_id"]
        self.model_kwargs = loader_config["model_kwargs"]
        try:
            # import in class level to avoid installing boto3
            import boto3

            boto3.__version__

            self._bedrock_client = aws_session.client(
                "bedrock-runtime", region_name=self._aws_region
            )

        except ImportError as e:
            raise ModuleNotFoundError(
                "Failed to import the 'boto3' Python package. "
                "Please install it by running `pip install boto3`."
            ) from e
        except Exception as e:
            raise ValueError(
                "Failed to load credentials for authenticating with the AWS client. "
                "Please ensure that the specified profile name contains valid credentials."
            ) from e

    def get_text_embedding(self, text: str):
        """
        Get the embedding representation of a given text.

        Args:
            text (str): The input text to be embedded.

        Returns:
            The embedding representation of the input text.
        """
        # replace newlines, which can negatively affect performance.
        text = text.replace(os.linesep, " ")

        # format input body for provider
        provider = self.model_id.split(".", maxsplit=1)[0]
        _model_kwargs = self.model_kwargs or {}
        input_body = {**_model_kwargs}
        if provider == "cohere":
            if "input_type" not in input_body:
                input_body["input_type"] = "search_document"
            input_body["texts"] = [text]
        else:
            # includes common provider == "amazon"
            input_body["inputText"] = text
        body = json.dumps(input_body)

        try:
            # invoke bedrock_client
            response = self._bedrock_client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )

            # format output based on provider
            response_body = json.loads(response.get("body").read())
            if provider == "cohere":
                return response_body.get("embeddings")[0]
            else:
                # includes common provider == "amazon"
                return response_body.get("embedding")
        except Exception as e:
            raise ValueError(f"Error raised by inference endpoint: {e}") from e
