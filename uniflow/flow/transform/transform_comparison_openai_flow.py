"""Paper Comparison OpenAI Model Flow Module."""

import re
from typing import Any, Dict, Sequence

from uniflow.constants import TRANSFORM
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.basic.expand_op import ExpandOp
from uniflow.op.basic.group_op import GroupOp
from uniflow.op.basic.reduce_op import ReduceOp
from uniflow.op.model.lm.model import JsonLmModel, LmModel
from uniflow.op.model.model_op import ModelOp
from uniflow.op.prompt import Context, PromptTemplate


class OpenAIComparisonFlow(Flow):
    """OpenAI Model Flow Class."""

    def __init__(
        self,
        prompt_template: PromptTemplate,
        model_config: Dict[str, Any],
    ) -> None:
        """OpenAI Model Flow Constructor.

        Args:
            prompt_template (PromptTemplate): Guided prompt template.
            model_config (Dict[str, Any]): Model config.
        """
        super().__init__()
        if model_config["response_format"]["type"] == "json_object":
            model = JsonLmModel(
                prompt_template=prompt_template,
                model_config=model_config,
            )
        else:
            model = LmModel(
                prompt_template=prompt_template,
                model_config=model_config,
            )

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
                for item in re.split(r"\n\s*\n", markdown_content[0].context)
                if item.strip()
            ],
        )

        # Add label
        label_prompt_template = PromptTemplate(
            instruction="""
            Assume you're a research scientist and are reading a research paper. You must classify it to be exactly one of the categories below, no extra wording. Classify the paragraph to be one of following catgeories:
            'label: 1-Abstract', 'label: 2-Introduction', 'label: 3-Background', 'label: 4-Approach', 'label: 5-Experiment or Result', 'label: 6-Conclusion or Future work'.
            If you are really unsure about it or don't have access to the content, just strictly return 'label: 3-Background'. Follow the example below.
            """,
            few_shot_prompt=[
                Context(
                    context="This study investigates the efficacy of mindfulness meditation in reducing stress levels among college students. A sample of 100 undergraduate students was randomly assigned to either a mindfulness meditation group or a control group. The mindfulness group underwent an eight-week mindfulness meditation program, while the control group received no intervention. Stress levels were measured using standardized self-report scales before and after the intervention period. Results indicate a significant reduction in perceived stress levels among participants in the mindfulness group compared to the control group. These findings suggest that mindfulness meditation may serve as an effective strategy for stress reduction among college students, highlighting its potential benefits for mental health promotion in academic settings.",
                    label="1-Abstract",
                ),
                Context(
                    context="Coral bleaching events pose a significant threat to the health and biodiversity of coral reef ecosystems, with climate change identified as a primary driver of these phenomena. This study utilizes satellite imagery and climate data to analyze the relationship between sea surface temperature anomalies and coral bleaching occurrences in the Great Barrier Reef (GBR) over the past two decades. Our results reveal a strong correlation between increased sea surface temperatures and the frequency and severity of coral bleaching events in the GBR region. Furthermore, projections based on climate models suggest a continued escalation of these events in the coming years under current emission scenarios. These findings underscore the urgent need for targeted conservation efforts and mitigation strategies to safeguard the long-term resilience of coral reef ecosystems in the face of climate change.",
                    label="1-Abstract",
                ),
                Context(
                    context="The global imperative to mitigate climate change and transition towards sustainable energy sources has propelled the rapid expansion of renewable energy generation. While renewable technologies offer immense potential for decarbonizing the electricity sector, their intermittent nature and spatial variability pose significant challenges to grid stability and reliability.",
                    label="2-Introduction",
                ),
                Context(
                    context="Author XYZ",
                    label="3-Background",
                ),
                Context(
                    context="XYZ University",
                    label="3-Background",
                ),
                Context(
                    context="Figure 3 illustrates the mean values of the dependent variable for each group, with the experimental group showing a noticeable improvement compared to the control group. Moreover, correlation analysis revealed a strong positive relationship between the treatment dosage and the improvement level (r = 0.78, p < 0.01), further substantiating the hypothesis. These results are consistent with the theoretical framework proposed, suggesting that the intervention directly contributes to the observed outcomes.",
                    label="5-Experiment or Result",
                ),
                Context(
                    context="In conclusion, the findings from this study provide substantial evidence supporting the hypothesis that the intervention significantly improves the outcome measures compared to the control. The statistical analysis, indicating both significance and a strong positive correlation between treatment dosage and effect size, underscores the potential of the intervention for practical applications. ",
                    label="6-Conclusion or Future work",
                ),
            ],
        )
        self._model_label = ModelOp(
            name="openai_model_label",
            model=LmModel(
                prompt_template=label_prompt_template,
                model_config=model_config,
            ),
        )

        # Summarize
        summary_prompt_template = PromptTemplate(
            instruction="""
            Assume you're a research scientist and are reading a research paper. 
            Please provide a detailed and comprehensive summary of each paragrph in the essay. Ignore insignificant and minor details such as emails.   
            """,
        )

        self._model_summary = ModelOp(
            name="openai_model_summarize",
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
                "label: 1-Abstract",
                "label: 2-Introduction",
                "label: 3-Background",
                "label: 4-Approach",
                "label: 5-Experiment or Result",
                "label: 6-Conclusion or Future work",
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
            Assume you're a research scientist and are reading two research papers. Ignore insignificant and minor details such as emails.
            Compare between paper A and paper B. Note their similarities and differences if applicable. 
            """,
        )

        self._model_compare = ModelOp(
            name="openai_model_compare_chunks",
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


class TransformComparisonOpenAIFlow(OpenAIComparisonFlow):
    """Transform Comparison OpenAI Flow Class."""

    TAG = TRANSFORM
