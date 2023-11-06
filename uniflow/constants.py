"""Flow constants"""
from uniflow.flow.flow import Flow
from uniflow.flow.flow_data_gen import DataGenFlow
from uniflow.flow.flow_data_gen_text import DataGenTextFlow
from uniflow.flow.flow_text_plus_data_gen import TextPlusDataGenFlow
from uniflow.flow.flow_self_instructed_gen import (
    SelfInstructedGenFlow,
    PDFSelfInstructedGenFlow,
)

flows_dict = {
    "flow": Flow,
    "flow_data_gen": DataGenFlow,
    "flow_data_gen_text": DataGenTextFlow,
    "flow_text_plus_data_gen": TextPlusDataGenFlow,
    "flow_self_instructed_gen": SelfInstructedGenFlow,
    "flow_pdf_self_instructed_gen": PDFSelfInstructedGenFlow,
}
