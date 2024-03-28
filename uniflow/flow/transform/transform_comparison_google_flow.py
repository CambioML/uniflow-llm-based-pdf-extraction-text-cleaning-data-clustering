"""Paper Comparison Google Model Flow Module."""

import re
from typing import Any, Dict, Sequence

from uniflow.constants import TRANSFORM
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.basic.expand_op import ExpandOp
from uniflow.op.basic.group_op import GroupOp
from uniflow.op.basic.reduce_op import ReduceOp
from uniflow.op.model.lm.model import LmModel
from uniflow.op.model.model_op import ModelOp
from uniflow.op.prompt import Context, PromptTemplate


class GoogleComparisonFlow(Flow):
    """Google Compariosn Flow Class."""

    def __init__(
        self,
        prompt_template: PromptTemplate,
        model_config: Dict[str, Any],
    ) -> None:
        """Google Compariosn Flow Constructor.

        Args:
            prompt_template (PromptTemplate): Guided prompt template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__()

        # Expand list of nodes to two or more nodes
        self._expand_from_papers = ExpandOp(
            name="expand_to_paper_node_from_nodes",
            fn=lambda x: [[x[0][i]] for i in range(len(x[0]))],
        )

        # Split into chunks
        self._expand_to_chunks = ExpandOp(
            name="split_to_chunks",
            fn=lambda markdown_content: [
                [Context(context=item.strip())]
                for item in re.split(r"\n\s*\n", markdown_content[0].Context)
                if item.strip()
            ],
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
        self._group = GroupOp(
            name="summaries_groupby_labels",
            fn=lambda labels, summaries: {
                label: [s for l, s in zip(labels, summaries) if l == label]
                for label in set(labels)
            },
            given_fixed_labels=[
                "1-Abstract",
                "2-Introduction",
                "3-Background",
                "4-Approach",
                "5-Experiment or Result",
                "6-Conclusion or Future work",
            ],
        )

        # Reduce pair chunks from each paper into list of nodes
        self._reduce_op = ReduceOp(
            name="reduce_to_pairs",
            fn=lambda list1, list2: [
                Context(context=f"paper A: {a.context}, paper B: {b.context}")
                for a, b in zip(list1, list2)
            ],
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
        paper1_node, paper2_node = self._expand_from_papers(nodes[0])

        paper1_node_chunks = self._expand_to_chunks(paper1_node)
        paper2_node_chunks = self._expand_to_chunks(paper2_node)

        paper1_node_chunks_labels = self._model_label(paper1_node_chunks)
        paper1_node_chunks_summaries = self._model_summary(paper1_node_chunks)

        paper2_node_chunks_labels = self._model_label(paper2_node_chunks)
        paper2_node_chunks_summaries = self._model_summary(paper2_node_chunks)

        paper1_node_grouped = self._group(
            paper1_node_chunks_labels, paper1_node_chunks_summaries
        )
        paper2_node_grouped = self._group(
            paper2_node_chunks_labels, paper2_node_chunks_summaries
        )

        combined_nodes = []
        for node_1, node_2 in zip(paper1_node_grouped, paper2_node_grouped):
            combined_nodes.append(self._reduce_op([(node_1, node_2)])[0])

        # TODO: add a model to fine fune overall comparison if needed

        return self._model_compare(combined_nodes)


class TransformComparisonGoogleFlow(GoogleComparisonFlow):
    """Transform Google Flow Class."""

    TAG = TRANSFORM
