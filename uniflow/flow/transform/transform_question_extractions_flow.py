"""Paper Comparison OpenAI Model Flow Module."""

import re
from typing import Any, Dict, Sequence

from uniflow.constants import TRANSFORM
from uniflow.flow.flow import Flow
from uniflow.node import Node
from uniflow.op.basic.expand_op import ExpandOp
from uniflow.op.model.lm.model import JsonLmModel, LmModel
from uniflow.op.model.model_op import ModelOp
from uniflow.op.prompt import Context, PromptTemplate


class OpenAIQuestionExtractionFlow(Flow):
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

        # # Expand list of nodes to two or more nodes
        # self._expand_from_papers = ExpandOp(
        #     name="expand_to_paper_node_from_nodes",
        #     fn=lambda x: [[x[0][i]] for i in range(len(x[0]))],
        # )

        # Split into chunks
        self._expand_to_chunks = ExpandOp(
            name="split_to_chunks",
            fn=lambda markdown_content: [
                [Context(context=item.strip())]
                for item in re.split(r"\n\s*\n", markdown_content[0].Context)
                if item.strip()
            ],
        )

        # Extract questions

        extract_question_prompt = PromptTemplate(
            instruction="""Assume you are a financial analyst. Your goal is use the contetxt to gain information to help you decide if you should long or short the stock. 
                        Generate as many questions and its corresponding answers as you can based on context. 
                        Following the format of the examples below to include the same context, question, and answer in the response.""",
            few_shot_prompt=[
                Context(
                    context="In 1948, Claude E. Shannon published A Mathematical Theory of\nCommunication (Shannon, 1948) establishing the theory of\ninformation. In his article, Shannon introduced the concept of\ninformation entropy for the first time. We will begin our journey here.",
                    question="Who published A Mathematical Theory of Communication in 1948?",
                    answer="Claude E. Shannon.",
                ),
                Context(
                    context="""Fears that Strong Results are Temporary: There are concerns that Q2 & Q3 2022’s
                    impressive results are from one-off windfalls and won’t be sustainable. The fear is that
                    merchants’ urgent needs for destocking during a period of overall weak Chinese
                    economic consumption, allowed PDD to command higher take rates (via increased
                    merchant advertising). Additionally, PDD skews toward lower tier cities, where users
                    were proportionally less exposed to COVID lock downs.
                    While these concerns are valid, we don’t believe it impairs PDD’s long-term thesis. The
                    company is increasing its mix of high-value / branded items, which is allowing it to court
                    higher-tier residents. Additionally we believe that with the 2nd largest consumer base,
                    merchants will be inclined to spend an increasing amount of their advertising budget to
                    reach these users. Pinduoduo is actively creating value for merchant’s businesses (and
                    thus expanding merchant’s margin profiles), as we’ll describe below. This incremental
                    value operates in a “closed-loop”, is then recaptured by the platform""",
                    question="How is Pinduoduo (PDD) attempting to mitigate the risk associated with its reliance on lower-tier cities?",
                    answer="Pinduoduo is increasing its mix of high-value and branded items to attract higher-tier residents, diversifying its user base beyond lower-tier cities.",
                ),
                Context(
                    context="""Fears that Strong Results are Temporary: There are concerns that Q2 & Q3 2022’s
                    impressive results are from one-off windfalls and won’t be sustainable. The fear is that
                    merchants’ urgent needs for destocking during a period of overall weak Chinese
                    economic consumption, allowed PDD to command higher take rates (via increased
                    merchant advertising). Additionally, PDD skews toward lower tier cities, where users
                    were proportionally less exposed to COVID lock downs.
                    While these concerns are valid, we don’t believe it impairs PDD’s long-term thesis. The
                    company is increasing its mix of high-value / branded items, which is allowing it to court
                    higher-tier residents. Additionally we believe that with the 2nd largest consumer base,
                    merchants will be inclined to spend an increasing amount of their advertising budget to
                    reach these users. Pinduoduo is actively creating value for merchant’s businesses (and
                    thus expanding merchant’s margin profiles), as we’ll describe below. This incremental
                    value operates in a “closed-loop”, is then recaptured by the platform""",
                    question="What is the impact of Pinduoduo's increasing mix of high-value/branded items on its merchant base?",
                    answer="This strategy attracts higher-tier residents, encouraging merchants to spend more on advertising and increasing their sales and margins.",
                ),
                Context(
                    context="""Fears that Strong Results are Temporary: There are concerns that Q2 & Q3 2022’s
                    impressive results are from one-off windfalls and won’t be sustainable. The fear is that
                    merchants’ urgent needs for destocking during a period of overall weak Chinese
                    economic consumption, allowed PDD to command higher take rates (via increased
                    merchant advertising). Additionally, PDD skews toward lower tier cities, where users
                    were proportionally less exposed to COVID lock downs.
                    While these concerns are valid, we don’t believe it impairs PDD’s long-term thesis. The
                    company is increasing its mix of high-value / branded items, which is allowing it to court
                    higher-tier residents. Additionally we believe that with the 2nd largest consumer base,
                    merchants will be inclined to spend an increasing amount of their advertising budget to
                    reach these users. Pinduoduo is actively creating value for merchant’s businesses (and
                    thus expanding merchant’s margin profiles), as we’ll describe below. This incremental
                    value operates in a “closed-loop”, is then recaptured by the platform""",
                    question="In what way does Pinduoduo create incremental value for merchants, and how is this value recaptured by the platform?",
                    answer="PDD enhances merchant margins through increased sales, encouraging reinvestment in advertising, creating a 'closed-loop' of value recapture.",
                ),
                Context(
                    context="""
                    Investor Metrics	FY23 Q1	FY23 Q2	FY23 Q3	FY23 Q4	FY24 Q1
                    Office Commercial products and cloud services revenue growth (y/y)	7% / 13%	7% / 14%	13% / 17%	12% / 14%	15% / 14%
                    Office Consumer products and cloud services revenue growth (y/y)	7% / 11%	(2)% / 3%	1% / 4%	3% / 6%	3% / 4%
                    Office 365 Commercial seat growth (y/y)	14%	12%	11%	11%	10%
                    Microsoft 365 Consumer subscribers (in millions)	65.1	67.7	70.8	74.9	76.7
                    Dynamics products and cloud services revenue growth (y/y)	15% / 22%	13% / 20%	17% / 21%	19% / 21%	22% / 21%
                    LinkedIn revenue growth (y/y)	17% / 21%	10% / 14%	8% / 11%	6% / 8%	8%
                    Growth rates include non-GAAP CC growth (GAAP % / CC %).
                    """,
                    question="How has the growth of Office Commercial products and cloud services revenue changed over the past five quarters?",
                    answer="The growth of Office Commercial products and cloud services revenue has increased from 7%/13%\in FY23 Q1 to 15%/14%\in FY24 Q1. The highest growth was recorded in FY23 Q3 at 13%/17%.",
                ),
                Context(
                    context="""
                    Investor Metrics	FY23 Q1	FY23 Q2	FY23 Q3	FY23 Q4	FY24 Q1
                    Office Commercial products and cloud services revenue growth (y/y)	7% / 13%	7% / 14%	13% / 17%	12% / 14%	15% / 14%
                    Office Consumer products and cloud services revenue growth (y/y)	7% / 11%	(2)% / 3%	1% / 4%	3% / 6%	3% / 4%
                    Office 365 Commercial seat growth (y/y)	14%	12%	11%	11%	10%
                    Microsoft 365 Consumer subscribers (in millions)	65.1	67.7	70.8	74.9	76.7
                    Dynamics products and cloud services revenue growth (y/y)	15% / 22%	13% / 20%	17% / 21%	19% / 21%	22% / 21%
                    LinkedIn revenue growth (y/y)	17% / 21%	10% / 14%	8% / 11%	6% / 8%	8%
                    Growth rates include non-GAAP CC growth (GAAP % / CC %).
                    """,
                    question="How did Office 365 Commercial seat growth change over the five quarters?",
                    answer="Office 365 Commercial seat growth decreased steadily from 14% in FY23 Q1 to 10%\in FY24 Q1.",
                ),
            ]
            # TODO: add more few_shot_prompts with tablur data
        )

        self._model_questions = ModelOp(
            name="openai_model_questions",
            model=LmModel(
                prompt_template=extract_question_prompt,
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

        paper1_node_chunks = self._expand_to_chunks(nodes[0])
        return self._model_questions(paper1_node_chunks)


class TransformQuestionExtractionOpenAIFlow(OpenAIQuestionExtractionFlow):
    """Transform Comparison OpenAI Flow Class."""

    TAG = TRANSFORM
