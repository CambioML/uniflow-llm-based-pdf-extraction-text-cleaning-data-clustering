from typing import Any, Dict

from uniflow.constants import TRANSFORM
from uniflow.model.flow.model_openai_flow import OpenAIModelFlow
from uniflow.schema import GuidedPrompt


class TransformOpenAIFlow(OpenAIModelFlow):
    tag = TRANSFORM

    def __init__(
        self,
        guided_prompt_template: GuidedPrompt,
        model_config: Dict[str, Any],
    ):
        super().__init__(
            guided_prompt_template=guided_prompt_template,
            model_config=model_config,
        )
