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


class OpenAINewsFeedFlow(Flow):
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

        # Another possible option is to pass in questions bank as a node and parse it into list of questions than call ReduceOp

        # Expand list of nodes to two or more nodes
        self._expand_from_papers = ExpandOp(
            name="expand_to_paper_node_from_nodes",
            fn=lambda x: [[x[0][i]] for i in range(len(x[0]))],
        )

        # Split into chunks (question node only)
        self._expand_to_chunks = ExpandOp(
            name="split_to_chunks",
            fn=lambda markdown_content: [
                [Context(context=item.strip())]
                for item in re.split(r"\n\s*\n", markdown_content[0].Context)
                if item.strip()
            ],
        )

        # question_banks = self._expand_from_papers(node)

        self._model_questions = []

        question_banks = prompt_template.instruction.split("\n")

        # Proposal 1: group question into catagory and classify news feed and then only ask specific question

        # TODO: have finer categories than just ["Company-Specific Information", "Market and Economic Analysis", "Governance", "Political Factors", "Other"]

        # label_prompt_template = PromptTemplate(
        #     instruction="""
        #     Assume you're an financial analyst working for institutional investors.
        #     You must classify the context to be exactly one of the labels below, no extra wording. Classify the paragraph to be one of following catgeories:
        #     'label: 1-Company-Specific Information', 'label: 2-Market and Economic Analysis', 'label: 3-Governance', 'label: 4-Political Factors', 'label: 5-Other'.
        #     If you are really unsure about it or don't have access to the content, just strictly return 'label: 5-Other'. Follow the example below.
        #     """,
        #     few_shot_prompt=[
        #         Context(
        #             context="What is the stock symbol for Apple?",
        #             label="1-Company-Specific Information",
        #         ),
        #         Context(
        #             context="How is Tesla currently valued compared to its expected future performance?",
        #             label="1-Company-Specific Information",
        #         ),
        #         Context(
        #             context="What factors influence the potential profitability of Rivian and the amount of investment spend into Amazon?",
        #             label="1-Company-Specific Information",
        #         ),
        #         Context(
        #             context="What new business line did Amazon recently launch?",
        #             label="1-Company-Specific Information",
        #         ),
        #         Context(
        #             context="What are some of the company-specific concerns contributing to the investment community's aversion to Chinese equities?",
        #             label="2-Market and Economic Analysis",
        #         ),
        #         Context(
        #             context="How has the U.S.-China trade war impacted Chinese tech stocks?",
        #             label="2-Market and Economic Analysis",
        #         ),
        #         Context(
        #             context="How do interest rate changes influence equity markets globally?",
        #             label="2-Market and Economic Analysis",
        #         ),
        #         Context(
        #             context="What is the outlook for global supply chains in the next five years?",
        #             label="2-Market and Economic Analysis",
        #         ),
        #         Context(
        #             context="How does Tesla approach executive compensation and incentives?",
        #             label="3-Governance",
        #         ),
        #         Context(
        #             context="What are the main principles guiding corporate governance at Google?",
        #             label="3-Governance",
        #         ),
        #         Context(
        #             context="How does Scale AI ensure compliance with regulatory requirements?",
        #             label="3-Governance",
        #         ),
        #         Context(
        #             context="How could the upcoming elections in the U.S. affect fiscal and trade policies?",
        #             label="4-Political Factors",
        #         ),
        #         Context(
        #             context="What impact do government regulations on data privacy have on technology companies?",
        #             label="4-Political Factors",
        #         ),
        #         Context(
        #             context="How does the political climate in Latin America affect commodity prices?",
        #             label="4-Political Factors",
        #         ),
        #         Context(
        #             context="How does consumer sentiment influence retail sales during the holiday season?",
        #             label="5-Other",
        #         ),
        #         Context(
        #             context="How is the gig economy shaping the future of work and employment?",
        #             label="5-Other",
        #         ),
        #         Context(
        #             context="How is the rise of telehealth transforming the healthcare industry?",
        #             label="5-Other",
        #         ),
        #         Context(
        #             context="What are the latest advancements in renewable energy technologies?",
        #             label="5-Other",
        #         ),
        #     ],
        # )

        # self._model_label = ModelOp(
        #     name="openai_model_label",
        #     model=LmModel(
        #         prompt_template=label_prompt_template,
        #         model_config=model_config,
        #     ),
        # )

        # TODO: potential weighting/count for questions needed each category controlled by users, maybe done in report generation level
        # Ex. only select top 5 questions from compnay specific category and 2 questions from political factor category

        # - Proposal 2: ask regarding what is the relevant question to the news before actually ask the question [and respond with question + answer]

        relevancy_instruction = """
        Given the question, consider if the question is relevant to the context. If so, generate an answer based on the context. If not, respond with question + 'answer: N/A'.
        Follow the example below to include both question and answer.
        """

        relevancy_few_shot_prompt = [
            Context(
                context="Apple continues to invest in research and development (R&D) to drive innovation across its product and service lines. The company is also focusing on sustainability initiatives, aiming to achieve carbon neutrality across its entire supply chain by 2030. Expansion of its services ecosystem and enhancement of customer engagement remain top priorities.",
                question="What are Apple's key strategic initiatives for the future?",
                answer="Apple is focusing on sustained innovation through increased investment in R&D, achieving carbon neutrality across its supply chain by 2030, expanding its services ecosystem, and enhancing customer engagement through high-quality products and services.",
            ),
            Context(
                context="Apple continues to invest in research and development (R&D) to drive innovation across its product and service lines. The company is also focusing on sustainability initiatives, aiming to achieve carbon neutrality across its entire supply chain by 2030. Expansion of its services ecosystem and enhancement of customer engagement remain top priorities.",
                question="How did the gross margin in Q2 2023 compare to Q2 2022?",
                answer="N/A",
            ),
            Context(
                context="Apple continues to invest in research and development (R&D) to drive innovation across its product and service lines. The company is also focusing on sustainability initiatives, aiming to achieve carbon neutrality across its entire supply chain by 2030. Expansion of its services ecosystem and enhancement of customer engagement remain top priorities.",
                question="What were Apple's Q2 2023 revenue and net income results?",
                answer="N/A",
            ),
            Context(
                context="Apple continues to invest in research and development (R&D) to drive innovation across its product and service lines. The company is also focusing on sustainability initiatives, aiming to achieve carbon neutrality across its entire supply chain by 2030. Expansion of its services ecosystem and enhancement of customer engagement remain top priorities.",
                question="What is the impact of the consumer wallet share gain strategy on revenue and earnings growth?",
                answer="N/A",
            ),
            Context(
                context="""
                Apple's Growth Drivers
                Strong iPhone Sales: Continued high demand for the iPhone 14 series, especially in key markets such as the U.S. and China.
                Expansion of Services Segment: Growth in services such as Apple Music, iCloud, and the App Store contributed significantly to revenue.
                Wearables and Accessories: Increased sales of Apple Watch and AirPods.
                Emerging Markets: Increased penetration in emerging markets, notably India and Southeast Asia.
                """,
                question="What is Apple’s strategy for growth in the services segment?",
                answer="Apple's strategy for growth in the services segment includes expanding offerings such as Apple Music, iCloud, and the App Store, as well as introducing new services to enhance customer engagement and loyalty.",
            ),
        ]

        self._model_questions_relevancy = []

        for question in question_banks:
            self._model_questions_relevancy.append(
                ModelOp(
                    name="openai_model_questions",
                    model=LmModel(
                        prompt_template=PromptTemplate(
                            instruction="Assume you are a financial analyst for institutional investors."
                            + question
                            + relevancy_instruction,
                            few_shot_prompt=relevancy_few_shot_prompt,
                        ),
                        model_config=model_config,
                    ),
                )
            )

    def run(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """

        answers = []

        for question_model in self._model_questions_relevancy:
            result = question_model(nodes)
            # print("DEBUG: ", result[0].value_dict)
            answers.append(result[0])

        return answers


class TransformNewsFeedOpenAIFlow(OpenAINewsFeedFlow):
    """Transform News Feed OpenAI Flow Class."""

    TAG = TRANSFORM
