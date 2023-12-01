"""Flow __init__ module."""
# this register all possible flow into FlowFactory through
# FlowFactory.register(cls.__name__, cls) in Flow __init_subclass__
from uniflow.flow.few_shot_model_flow import FewShotModelFlow  # noqa: F401
from uniflow.flow.flow import LinearFlow  # noqa: F401
from uniflow.flow.model_flow import (HuggingFaceModelFlow,  # noqa: F401
                                     ModelFlow, OpenAIJsonModelFlow)
