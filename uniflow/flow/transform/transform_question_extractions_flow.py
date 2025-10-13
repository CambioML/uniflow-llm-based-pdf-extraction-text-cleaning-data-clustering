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
        # TODO: only need mutiple questions or list of questions, not answers

        extract_question_prompt = PromptTemplate(
            instruction="""Assume you are a financial analyst. Your goal is use the contetxt to gain information to help you decide if you should long or short the stock. 
                        Generate as a list of insightful questions based on context.
                        Following the format of the examples to include question list below.""",
            few_shot_prompt=[
                Context(
                    context="""Fears that Strong Results are Temporary: There are concerns that Q2 & Q3 2022's
                    impressive results are from one-off windfalls and won't be sustainable. The fear is that
                    merchants' urgent needs for destocking during a period of overall weak Chinese
                    economic consumption, allowed PDD to command higher take rates (via increased
                    merchant advertising). Additionally, PDD skews toward lower tier cities, where users
                    were proportionally less exposed to COVID lock downs.
                    While these concerns are valid, we don't believe it impairs PDD's long-term thesis. The
                    company is increasing its mix of high-value / branded items, which is allowing it to court
                    higher-tier residents. Additionally we believe that with the 2nd largest consumer base,
                    merchants will be inclined to spend an increasing amount of their advertising budget to
                    reach these users. Pinduoduo is actively creating value for merchant's businesses (and
                    thus expanding merchant's margin profiles), as we'll describe below. This incremental
                    value operates in a “closed-loop”, is then recaptured by the platform""",
                    question=[
                        "How is Pinduoduo (PDD) attempting to mitigate the risk associated with its reliance on lower-tier cities?",
                        "What is the impact of Pinduoduo's increasing mix of high-value/branded items on its merchant base?",
                        "In what way does Pinduoduo create incremental value for merchants, and how is this value recaptured by the platform?",
                    ],
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
                    question=[
                        "How has the growth of Office Commercial products and cloud services revenue changed over the past five quarters?",
                        "How did Office 365 Commercial seat growth change over the five quarters?",
                    ],
                ),
                Context(
                    context="""
                    When looking at the markets that have the most potential for expansion, there are two
                    that jump out as having the most promise. These two markets present an opportunity
                    for Tesla to go even further. As was noted earlier, Europe is the region that saw the
                    biggest quantity of units sold which is the market that represents the largest percentage
                    of the global market. Both of these markets are far from being saturated, which means
                    there is a significant possibility here. Tesla is well aware of the situation and has already
                    established a Giga factory in Berlin in addition to the one it has established in Shanghai.
                    In addition, Tesla is planning to establish a number of additional facilities, the most of
                    which will be situated in Asia, with the goals of satisfying the demand in the market
                    and bringing down costs. A well-funded company could develop a new electric vehicle
                    (EV) from scratch and move it into production within 3 to 5 years, by spending $1 - 2
                    billion of capital for design, development, and manufacturing.
                    """,
                    question=[
                        "What is Tesla's current market share in Europe and Asia, and how does it compare to other major players in these regions?",
                        "What are the major barriers to entry for new competitors in these markets, and how is Tesla positioning itself to maintain a competitive edge?",
                    ],
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
