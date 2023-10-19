"""Flow constants"""
from uniflow.flow.flow import Flow
from uniflow.flow.flow_data_gen import DataGenFlow
from uniflow.flow.flow_data_gen_text import DataGenTextFlow
from uniflow.flow.flow_text_plus_data_gen import TextPlusDataGenFlow

flows_dict = {
    "flow": Flow,
    "low_data_gen": DataGenFlow,
    "flow_data_gen_text": DataGenTextFlow,
    "flow_text_plus_data_gen": TextPlusDataGenFlow,
}
