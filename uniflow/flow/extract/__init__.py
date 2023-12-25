"""Model __init__ Module."""

# this register all possible model server into ModelServerFactory through
# ModelServerFactory.register(cls.__name__, cls) in AbsModelServer
# __init_subclass__

from uniflow.flow.extract.extract_ipynb_flow import ExtractIpynbFlow
from uniflow.flow.extract.extract_mardown_flow import ExtractMarkdownFlow
from uniflow.flow.extract.extract_pdf_flow import ExtractPDFFlow
from uniflow.flow.extract.extract_txt_flow import ExtractTxtFlow

__all__ = [
    "ExtractIpynbFlow",
    "ExtractMarkdownFlow",
    "ExtractPDFFlow",
    "ExtractTxtFlow",
]
