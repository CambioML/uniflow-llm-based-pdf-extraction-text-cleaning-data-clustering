"""Flow __init__ module."""
# this register all possible flow into FlowFactory through
# FlowFactory.register(cls.__name__, cls) in Flow __init_subclass__

from uniflow.extract.extract_txt_flow import ExtractTxtFlow  # noqa: F401
