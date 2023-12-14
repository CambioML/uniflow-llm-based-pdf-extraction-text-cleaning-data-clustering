"""Flow __init__ module."""
# this register all possible flow into FlowFactory through
# FlowFactory.register(cls.__name__, cls) in Flow __init_subclass__
from uniflow.extract.flow.extract_txt_flow import ExtractTxtFlow  # noqa: F401, F403
from uniflow.transform.flow.transform_huggingface_flow import (  # noqa: F401, F403
    TransformHuggingFaceFlow,
)
from uniflow.transform.flow.transform_lmqg_flow import (  # noqa: F401, F403
    TransformLMQGFlow,
)
from uniflow.transform.flow.transform_openai_flow import (  # noqa: F401, F403
    TransformOpenAIFlow,
)

__all__ = [
    "ExtractTxtFlow",
    "TransformOpenAIFlow",
    "TransformLMQGFlow",
    "TransformHuggingFaceFlow",
]
