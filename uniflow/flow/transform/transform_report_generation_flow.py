"""Paper Comparison OpenAI Model Flow Module."""

import re
from typing import Any, Dict, Sequence

from uniflow.constants import TRANSFORM
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.basic.expand_op import ExpandOp
from uniflow.op.basic.group_op import GroupOp
from uniflow.op.model.lm.model import JsonLmModel, LmModel
from uniflow.op.model.model_op import ModelOp
from uniflow.op.prompt import Context, PromptTemplate


class OpenAIReportGenerationFlow(Flow):
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
        # TODO: Refactoring needed to make model_op output Context format. Need to keep it in Context format and only convert back to dictionary format before exiting Flow
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
                for item in re.split(r"\n", markdown_content[0].context)
                if item.strip()
            ],
        )

        # Proposal 1: group question into catagory and classify news feed and then only ask specific question

        # TODO: have finer categories than just ["Company-Specific Information", "Market and Economic Analysis", "Governance", "Political Factors", "Other"]

        label_prompt_template = PromptTemplate(
            instruction="""
            Assume you're an financial analyst working for institutional investors. 
            You must classify the context to be exactly one of the labels below, no extra wording. Classify the paragraph to be one of following catgeories:
            'label: 1-Company-Specific Information', 'label: 2-Market and Economic Analysis', 'label: 3-Governance', 'label: 4-Political Factors', 'label: 5-Other'.  
            If you are really unsure about it or don't have access to the content, just strictly return 'label: 5-Other'. Follow the example below.
            """,
            few_shot_prompt=[
                Context(
                    context="What is the stock symbol for Apple?",
                    label="1-Company-Specific Information",
                ),
                Context(
                    context="How is Tesla currently valued compared to its expected future performance?",
                    label="1-Company-Specific Information",
                ),
                Context(
                    context="What factors influence the potential profitability of Rivian and the amount of investment spend into Amazon?",
                    label="1-Company-Specific Information",
                ),
                Context(
                    context="What new business line did Amazon recently launch?",
                    label="1-Company-Specific Information",
                ),
                Context(
                    context="What are some of the company-specific concerns contributing to the investment community's aversion to Chinese equities?",
                    label="2-Market and Economic Analysis",
                ),
                Context(
                    context="How has the U.S.-China trade war impacted Chinese tech stocks?",
                    label="2-Market and Economic Analysis",
                ),
                Context(
                    context="How do interest rate changes influence equity markets globally?",
                    label="2-Market and Economic Analysis",
                ),
                Context(
                    context="What is the outlook for global supply chains in the next five years?",
                    label="2-Market and Economic Analysis",
                ),
                Context(
                    context="How does Tesla approach executive compensation and incentives?",
                    label="3-Governance",
                ),
                Context(
                    context="What are the main principles guiding corporate governance at Google?",
                    label="3-Governance",
                ),
                Context(
                    context="How does Scale AI ensure compliance with regulatory requirements?",
                    label="3-Governance",
                ),
                Context(
                    context="How could the upcoming elections in the U.S. affect fiscal and trade policies?",
                    label="4-Political Factors",
                ),
                Context(
                    context="What impact do government regulations on data privacy have on technology companies?",
                    label="4-Political Factors",
                ),
                Context(
                    context="How does the political climate in Latin America affect commodity prices?",
                    label="4-Political Factors",
                ),
                Context(
                    context="How does consumer sentiment influence retail sales during the holiday season?",
                    label="5-Other",
                ),
                Context(
                    context="How is the gig economy shaping the future of work and employment?",
                    label="5-Other",
                ),
                Context(
                    context="How is the rise of telehealth transforming the healthcare industry?",
                    label="5-Other",
                ),
                Context(
                    context="What are the latest advancements in renewable energy technologies?",
                    label="5-Other",
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

        # Group summaries by label
        self._group = GroupOp(
            name="summaries_groupby_labels",
            preprocss_fn=lambda nodes_1, nodes_2: [
                (
                    node_label.value_dict["response"][0],
                    node_summary.value_dict["response"][0],
                )
                for node_label, node_summary in zip(nodes_1, nodes_2)
            ],
            fn=lambda labels, summaries: {
                label: [s for l, s in zip(labels, summaries) if l == label]
                for label in set(labels)
            },
            given_fixed_labels=[
                "label: 1-Company-Specific Information",
                "label: 2-Market and Economic Analysis",
                "label: 3-Governance",
                "label: 4-Political Factors",
                "label: 5-Other",
            ],
        )

        # TODO: potential weighting/count for questions needed each category controlled by users, maybe done in report generation level
        # Ex. only select top 5 questions from compnay specific category and 2 questions from political factor category

        # TODO: add a summary model in the end to polish results

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        question_node, answer_node = self._expand_from_papers(nodes[0])

        question_node_chunks = self._expand_to_chunks(question_node)

        # for node in question_node_chunks:
        #     print("debug: ", node.value_dict)

        answer_node_chunks = self._expand_to_chunks(answer_node)

        question_node_labels = self._model_label(question_node_chunks)

        # for node in question_node_labels:
        #     print("debug: ", node.value_dict)

        # raise Exception("checkpoint")

        answer_node_grouped = self._group(question_node_labels, answer_node_chunks)

        return answer_node_grouped

        # generate report by section on answer_node_grouped


class TransformReportGenerationOpenAIFlow(OpenAIReportGenerationFlow):
    """Transform Report Generation OpenAI Flow Class."""

    TAG = TRANSFORM
