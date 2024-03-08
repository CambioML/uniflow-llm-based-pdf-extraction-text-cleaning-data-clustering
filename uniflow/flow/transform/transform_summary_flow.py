"""Paper Comparison Flow Module."""

from typing import Any, Dict, Sequence

from uniflow.constants import TRANSFORM
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.model.lm.model import LmModel
from uniflow.op.model.model_op import ModelOp
from uniflow.op.prompt import PromptTemplate
from uniflow.op.prompt import Context

from uniflow.op.basic.expand_op import ExpandOp
from uniflow.op.basic.reduce_op import ReduceOp
from uniflow.op.basic.group_op import GroupOp


class GoogleModelFlow(Flow):
    """Google Model Flow Class."""

    def __init__(
        self,
        prompt_template: PromptTemplate,
        model_config: Dict[str, Any],
    ) -> None:
        """Google Model Flow Constructor.

        Args:
            prompt_template (PromptTemplate): Guided prompt template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__()
        self._model_op = ModelOp(
            name="google_model_op",
            model=LmModel(
                prompt_template=prompt_template,
                model_config=model_config,
            ),
        )

        # Expand list of nodes to two or more nodes 
        self.expand0_fn = lambda x: [[x[0][i]] for i in range(len(x[0]))]
        self._expand0_op = ExpandOp(
            name="google_model_exapnd0",
            fn=self.expand0_fn
        )

        # Split into chunks
        self.expand1_fn = lambda d: [[Context(context=item.strip())] for item in d[0].Context.split('\n') if item.strip()]

        self._expand1_op = ExpandOp(
            name="google_model_expand1",
            fn=self.expand1_fn
        )

        # Add label
        label_prompt_template = PromptTemplate(
            instruction="""
            Assume you're a research scientist and are reading a research paper. Classify the paragraph to be one of following catgeories:
            "1-Abstract", "2-Introduction", "3-Background", "4-Approach", "5-Experiment or Result", "6-Conclusion or Future work"
            """,
        )

        self._model_label = ModelOp(
            name="google_model_label",
            model=LmModel(
                prompt_template=label_prompt_template,
                model_config=model_config,
            ),
        )

        # Summarize 
        summary_prompt_template = PromptTemplate(
            instruction="""
            Assume you're a research scientist and are reading a research paper. 
            Please provide a detailed and comprehensive summary of each paragrph in the essay.   
            """,
        )

        self._model_summary = ModelOp(
            name="google_model_summary",
            model=LmModel(
                prompt_template=summary_prompt_template,
                model_config=model_config,
            ),
        )

        # Group summaries by label
        self.group_fn = lambda labels, summaries: {label: [s for l, s in zip(labels, summaries) if l == label] for label in set(labels)}
        self.preprocess_fn = lambda nodes_1, nodes_2: [(node_label.value_dict['response'][0], node_summary.value_dict['response'][0])
                                                       for node_label, node_summary in zip(nodes_1, nodes_2)]

        self._group = GroupOp(
            name="google_model_groupby",
            preprocss_fn=self.preprocess_fn,
            fn=self.group_fn
        )

        # Reduce pair chunks from each paper into list of nodes
        self.reduce_fn = lambda list1, list2: [Context(context=f"paper A: {a.context}, paper B: {b.context}") for a, b in zip(list1, list2)]
        self._reduce_op = ReduceOp(
            name="google_model_reduce",
            fn=self.reduce_fn
        ) 

        # Compare
        compare_prompt_template = PromptTemplate(
            instruction="""
            Assume you're a research scientist and are reading two research papers.
            Compare between paper A and paper B. Note their similarities and differences if applicable. 
            """,
        )

        self._model_compare = ModelOp(
            name="google_model_compare",
            model=LmModel(
                prompt_template=compare_prompt_template,
                model_config=model_config,
            ),
        )


    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """

        paper1_node, paper2_node = self._expand0_op(nodes[0])

        p3_op_nodes_1 = self._expand1_op(paper1_node)
        p3_op_nodes_2 = self._expand1_op(paper2_node)

        label_nodes_1 = self._model_label(p3_op_nodes_1)
        summary_nodes_1 = self._model_summary(p3_op_nodes_1)

        label_nodes_2 = self._model_label(p3_op_nodes_2)
        summary_nodes_2 = self._model_summary(p3_op_nodes_2)

        p4_op_nodes_1 = self._group(label_nodes_1, summary_nodes_1)
        p4_op_nodes_2 = self._group(label_nodes_2, summary_nodes_2)

        p5_op_nodes = []
        for node_1, node_2 in zip(p4_op_nodes_1, p4_op_nodes_2):
            p5_op_nodes.append(self._reduce_op([(node_1, node_2)])[0])
        
        p6_op_nodes = self._model_compare(p5_op_nodes)

        # TODO: add a model to fine fune overall comparison if needed  
        
        return p6_op_nodes


class TransformSummaryGoogleFlow(GoogleModelFlow):
    """Transform Google Flow Class."""  

    TAG = TRANSFORM
