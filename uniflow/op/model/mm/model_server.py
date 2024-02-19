"""All Multimodal Model Servers."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from uniflow.op.model.model_config import GoogleModelConfig
from uniflow.op.model.model_server import AbsModelServer
from uniflow.op.prompt import PromptTemplate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GoogleMultiModalModelServer(AbsModelServer):
    """Google MultiModal Model Server Class.

    TODO: this is a duplication of GoogleModelServer in lm/model_server.py.
    """

    def __init__(
        self, prompt_template: PromptTemplate, model_config: Dict[str, Any]
    ) -> None:

        try:
            import google.generativeai as genai  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Please install google-generativeai. You can use `pip install google-generativeai` to install it."
            ) from exc
        super().__init__(prompt_template, model_config)

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        self._model_config = GoogleModelConfig(**self._model_config)
        self._client = genai.GenerativeModel(model_name=self._model_config.model_name)
        self._generation_config = genai.types.GenerationConfig(
            # Only one candidate for now.
            candidate_count=self._model_config.candidate_count,
            max_output_tokens=self._model_config.max_output_tokens,
            temperature=self._model_config.temperature,
            top_k=self._model_config.top_k,
            top_p=self._model_config.top_p,
        )

    def _preprocess(self, data: List[str]) -> List[str]:
        """Preprocess data.

        Args:
            data (List[str]): Data to preprocess.

        Returns:
            List[str]: Preprocessed data.
        """
        return data

    def _make_api_call(self, data: str) -> str:
        """Helper method to make API call.

        Args:
            data (str): Data to run.

        Returns:
            str: Output data.
        """
        return self._client.generate_content(
            contents=data,
            generation_config=self._generation_config,
        )

    def _postprocess(self, data: str) -> List[str]:
        return [c.text for d in data for c in d.parts]

    def __call__(self, data: List[str]) -> List[str]:
        """Run model with ThreadPoolExecutor.

        Args:
            data (List[str]): Data to run.

        Returns:
            List[str]: Output data.
        """
        data = self._preprocess(data)

        # use ThreadPoolExecutor to parallelize API calls
        with ThreadPoolExecutor(max_workers=self._model_config.num_thread) as executor:
            futures = [executor.submit(self._make_api_call, d) for d in data]
            inference_data = [future.result() for future in futures]

        data = self._postprocess(inference_data)
        return data
