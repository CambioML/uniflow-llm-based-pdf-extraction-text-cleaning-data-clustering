"""Extract, Transform and Pipeline config module."""

from dataclasses import dataclass, field
from typing import Dict, Optional

from uniflow import Context, PromptTemplate
from uniflow.op.extract.split.constants import (
    MARKDOWN_HEADER_SPLITTER,
    PARAGRAPH_SPLITTER,
)
from uniflow.op.model.model_config import (
    BedrockModelConfig,
    HuggingfaceModelConfig,
    LMQGModelConfig,
    ModelConfig,
    NougatModelConfig,
    OpenAIModelConfig,
)

###########################################################
#                   All Extract Configs                   #
###########################################################


@dataclass
class ExtractConfig:
    """Extract Config Class."""

    flow_name: str
    num_thread: int = 1
    model_config: Optional[ModelConfig] = None
    splitter: Optional[str] = None


@dataclass
class ExtractTxtConfig(ExtractConfig):
    """Extract Txt Config Class."""

    flow_name: str = "ExtractTxtFlow"


@dataclass
class ExtractPDFConfig(ExtractConfig):
    """Nougat Config Class."""

    flow_name: str = "ExtractPDFFlow"
    model_config: ModelConfig = field(default_factory=NougatModelConfig())
    splitter: str = PARAGRAPH_SPLITTER


@dataclass
class ExtractMarkdownConfig(ExtractConfig):
    """Extract Markdown Config Class."""

    flow_name: str = "ExtractMarkdownFlow"
    splitter: str = MARKDOWN_HEADER_SPLITTER


@dataclass
class ExtractIpynbConfig(ExtractConfig):
    """Extract ipynb Config Class."""

    flow_name: str = "ExtractIpynbFlow"


###########################################################
#                   All Transform Configs                 #
###########################################################


@dataclass
class TransformConfig:
    """Transform Base Config Class."""

    flow_name: str
    model_config: ModelConfig = field(default_factory=ModelConfig)
    num_thread: int = 1
    prompt_template: PromptTemplate = field(
        default_factory=lambda: PromptTemplate(
            instruction="""
            Generate one question and its corresponding answer based on the last context in the last
            example. Follow the format of the examples below to include context, question, and answer in the response
            """,
            few_shot_prompt=[
                Context(
                    context="The quick brown fox jumps over the lazy black dog.",
                    question="What is the color of the fox?",
                    answer="brown.",
                ),
                Context(
                    context="The quick brown fox jumps over the lazy black dog.",
                    question="What is the color of the dog?",
                    answer="black.",
                ),
            ],
        )
    )


@dataclass
class TransformOpenAIConfig(TransformConfig):
    """Transform OpenAI Config Class."""

    flow_name: str = "TransformOpenAIFlow"
    model_config: ModelConfig = field(default_factory=OpenAIModelConfig())


@dataclass
class TransformHuggingFaceConfig(TransformConfig):
    """Transform Hugging Face Config Class."""

    flow_name: str = "TransformHuggingFaceFlow"
    model_config: ModelConfig = field(default_factory=HuggingfaceModelConfig())


@dataclass
class TransformLMQGConfig(TransformConfig):
    """Transform LMQG Config Class."""

    flow_name: str = "TransformLMQGFlow"
    prompt_template: PromptTemplate = field(
        default_factory=lambda: PromptTemplate(instruction="", few_shot_prompt=[])
    )
    model_config: ModelConfig = field(default_factory=LMQGModelConfig())


@dataclass
class TransformCopyConfig(TransformConfig):
    """Transform Linear Config Class."""

    flow_name: str = "TransformCopyFlow"
    prompt_template: PromptTemplate = field(
        default_factory=lambda: PromptTemplate(instruction="", few_shot_prompt=[])
    )
    model_config: ModelConfig = field(default_factory=lambda: {})

class TransformForClassificationOpenAIGPT3p5Config(TransformConfig):
    flow_name: str = "TransformOpenAIFlow"
    model_config: ModelConfig = field(
        default_factory=lambda: OpenAIModelConfig(
            model_name="gpt-3.5-turbo-1106",
            model_server="OpenAIModelServer",
            num_call=1,
            temperature=0,
            response_format={"type": "text"},
        )
    )
    prompt_template: PromptTemplate = field(
        default_factory=lambda: PromptTemplate(
            instruction="""
            Does the text mention any cutting-edged technology applications, any new technology methods, or any new area of innovations? If yes, return the names of each technology in a list of strings as the answer. If no, return an empty list.
            """,
            few_shot_prompt=[
                Context(
                    context="Our new business wins are supported by our product leadership strategy of bringing new product to market that provides value for our customers, such as market-leading 500 bar GDi technology, helping customers improve efficiency, reduce emissions and lower costs leveraging our GDi technology and capital to provide a value-focused solution for our off-highway diesel applications and hydrogen ICE that differentiates us from our competition. We're helping our customers move towards carbon neutral and carbon-free fuels with solutions using ethanol, biofuels and hydrogen, as it's our view that a liquefied or gaseous fuel is going to be a key element of our journey to carbon neutrality.",
                    answer=["500 bar GDi technology", "carbon neutral"]
                ),
                    Context(
                    context="The Eiffel Tower, located in Paris, France, is one of the most famous landmarks in the world. It was constructed in 1889 and stands at a height of 324 meters.",
                    answer=[],
                ),
            ],
        )
    )

@dataclass
class TransformForClusteringOpenAIGPT4Config:
    flow_name: str = "TransformOpenAIFlow"
    model_config: ModelConfig = field(
        default_factory=lambda: OpenAIModelConfig(
            model_name="gpt-4",
            model_server="OpenAIModelServer",
            num_call=1,
            temperature=0,
            response_format={"type": "json_object"}
        )
    )
    prompt_template: PromptTemplate = field(
        default_factory=lambda: PromptTemplate(
            instruction="""
                As an expert in cutting-edge technologies, your task is to analyze a given list of technology-related terms. Your goal is to cluster these terms into groups based on their semantic similarities. Each group represents a unique category or 'signal' of technology. You will return your analysis as a dictionary. In this dictionary, each key is a 'signal', representing a specific category, and the associated value is a list of technology terms that belong to that category based on their semantic meaning.
            """,
            few_shot_prompt = [
                Context(
                    context=["artificial intelligence", "AI", "500 bar GDi technology", "ML", "500 bar GDi", "machine learning"],
                    answer={
                        "500_BAR_GDI": ["500 bar GDi technology", "500 bar GDi"],
                        "AIML": ["artificial intelligence", "AI", "ML", "machine learning"],
                    }
                ),
                Context(
                    context=["cryptocurrency", "blockchain", "Bitcoin", "Ethereum", "digital currency", "crypto mining"],
                    answer={
                        "CRYPTO_CURRENCY": ["cryptocurrency", "Bitcoin", "Ethereum", "digital currency"],
                        "BLOCKCHAIN_TECH": ["blockchain", "crypto mining"],
                    },
                ),
            ]
        )
    )

###########################################################
#                   All AutoRater Config                  #
###########################################################
@dataclass
class RaterConfig:
    """Rater Config Class."""

    flow_name: str = "RaterFlow"
    model_config: ModelConfig = field(default_factory=ModelConfig)
    label2score: Dict[str, float] = field(default_factory=dict)
    prompt_template: PromptTemplate = field(default_factory=PromptTemplate)
    num_thread: int = 1

    def __post_init__(self):
        """Post-initialization to perform label check."""
        # Add label to the end of instruction to help produce more consistent response label.
        # If {label_list} {label2score} not in prompt_template, it won't cause error
        self.prompt_template.instruction = str(self.prompt_template.instruction).format(
            label_list=str(list(self.label2score.keys())),
            label2score=str(list(self.label2score.items())),
        )

        incompatible_labels = self.check_labels()
        unexpected_labels = incompatible_labels["unexpected_labels"]
        missing_labels = incompatible_labels["missing_labels"]
        if unexpected_labels:
            raise ValueError(
                "Inconsistent labels found in prompt_template examples, "
                f"example label {unexpected_labels} not in label2score has keys {list(self.label2score.keys())}",
            )
        if missing_labels:
            print(f"The label2score label {missing_labels} not in example label.")

    def check_labels(self) -> Dict[str, list]:
        """
        Check if every label in the prompt_template's examples is a key in label2score.

        Returns:
            Dict: Incompatible Keys, fields:
                missing_labels (List[str]): labels in label2score but not in examples, this may cause performance loss.
                unexpected_labels (List[str]): labels in examples but not in label2score, this cause ValueError.
        """
        example_labels = set()
        label2score_labels = set()
        # Check if prompt_template has examples
        if self.prompt_template.few_shot_prompt:
            for example in self.prompt_template.few_shot_prompt:
                example_labels.add(example.label)
            label2score_labels = set(self.label2score.keys())
        missing_labels = label2score_labels - example_labels
        unexpected_labels = example_labels - label2score_labels
        return {
            "missing_labels": list(missing_labels),
            "unexpected_labels": list(unexpected_labels),
        }


@dataclass
class RaterForClassificationOpenAIGPT4Config(RaterConfig):
    """
    The configuration primarily focuses on setting up the parameters for utilizing GPT-4 to evaluate the
    correctness of answers in relation to given questions and contexts.

    Attributes:
        flow_name (str): Name of the rating flow, default is "RaterFlow".
        model_config (ModelConfig): Configuration for the GPT-4 model. Includes model name ("gpt-4"),
                                    the server ("OpenAIModelServer"), number of calls (1), temperature (0),
                                    and the response format (plain text).
        label2score (Dict[str, float]): Mapping of labels to scores, default is {"Yes": 1.0, "No": 0.0}.
        prompt_template (PromptTemplate): Template for guided prompts used in rating. Includes instructions
                                               for rating, along with examples that detail the context, question,
                                               answer, label, and explanation for each case.
    """

    model_config: ModelConfig = field(
        default_factory=lambda: OpenAIModelConfig(
            model_name="gpt-4",
            model_server="OpenAIModelServer",
            num_call=1,
            temperature=0,
            response_format={"type": "text"},
        )
    )
    label2score: Dict[str, float] = field(
        default_factory=lambda: {"Yes": 1.0, "No": 0.0}
    )
    prompt_template: PromptTemplate = field(
        default_factory=lambda: PromptTemplate(
            instruction="""
            Evaluate the appropriateness of a given answer based on the question and the context.
            There are few examples below, consisting of context, question, answer, explanation and label.
            If answer is appropriate, you should give a higher score and vise versa. Check label to score dictionary: {label2score}.
            Your response should only focus on the unlabeled sample, including two fields: explanation and label (one of {label_list}).
            """,
            few_shot_prompt=[
                Context(
                    context="The Eiffel Tower, located in Paris, France, is one of the most famous landmarks in the world. It was constructed in 1889 and stands at a height of 324 meters.",
                    question="When was the Eiffel Tower constructed?",
                    answer="The Eiffel Tower was constructed in 1889.",
                    explanation="The context explicitly mentions that the Eiffel Tower was constructed in 1889, so the answer is correct.",
                    label="Yes",
                ),
                Context(
                    context="Photosynthesis is a process used by plants to convert light energy into chemical energy. This process primarily occurs in the chloroplasts of plant cells.",
                    question="Where does photosynthesis primarily occur in plant cells?",
                    answer="Photosynthesis primarily occurs in the mitochondria of plant cells.",
                    explanation="The context mentions that photosynthesis primarily occurs in the chloroplasts of plant cells, so the answer is incorrect.",
                    label="No",
                ),
            ],
        )
    )


@dataclass
class RaterForClassificationOpenAIGPT3p5Config(RaterConfig):
    """
    The configuration primarily focuses on setting up the parameters for utilizing GPT-3.5 to evaluate the
    correctness of answers in relation to given questions and contexts.
    Attributes:
        flow_name (str): Name of the rating flow, default is "RaterFlow".
        model_config (ModelConfig): Configuration for the GPT-3.5 model. Includes model name ("gpt-3.5-turbo-1106"),
                                    the server ("OpenAIModelServer"), number of calls (1), temperature (0),
                                    and the response format (plain text).
        label2score (Dict[str, float]): Mapping of labels to scores, default is {"Yes": 1.0, "No": 0.0}.
        prompt_template (PromptTemplate): Template for guided prompts used in rating. Includes instructions
                                               for rating, along with examples that detail the context, question,
                                               answer, label, and explanation for each case.
    """

    model_config: ModelConfig = field(
        default_factory=lambda: OpenAIModelConfig(
            model_name="gpt-3.5-turbo-1106",
            model_server="OpenAIModelServer",
            num_call=1,
            temperature=0,
            response_format={"type": "text"},
        )
    )
    label2score: Dict[str, float] = field(
        default_factory=lambda: {"Yes": 1.0, "No": 0.0}
    )
    prompt_template: PromptTemplate = field(
        default_factory=lambda: PromptTemplate(
            instruction="""
            # Task: Evaluate the appropriateness of a given answer based on a provided context and question.
            ## Input:
            1. context: A brief text containing key information.
            2. question: A query related to the context, testing knowledge that can be inferred or directly obtained from it.
            3. answer: A response to the question.
            ## Evaluation Criteria: If answer is appropriate, you should give a higher score and vise versa. Check label to score dictionary: {label2score}.
            ## Response Format: Your response should only include two fields below:
            1. explanation: Reasoning behind your judgment, explaining why the answer is appropriate or not.
            2. label: Your judgment (one of {label_list}).
            ## Note: Use the below example only for demonstration, do not include in the final response.
            """,
            few_shot_prompt=[
                Context(
                    context="The Eiffel Tower, located in Paris, France, is one of the most famous landmarks in the world. It was constructed in 1889 and stands at a height of 324 meters.",
                    question="When was the Eiffel Tower constructed?",
                    answer="The Eiffel Tower was constructed in 1889.",
                    explanation="The context explicitly mentions that the Eiffel Tower was constructed in 1889, so the answer is correct.",
                    label="Yes",
                ),
                Context(
                    context="Photosynthesis is a process used by plants to convert light energy into chemical energy. This process primarily occurs in the chloroplasts of plant cells.",
                    question="Where does photosynthesis primarily occur in plant cells?",
                    answer="Photosynthesis primarily occurs in the mitochondria of plant cells.",
                    explanation="The context mentions that photosynthesis primarily occurs in the chloroplasts of plant cells, so the answer is incorrect.",
                    label="No",
                ),
            ],
        )
    )


@dataclass
class RaterForClassificationBedrockClaudeConfig(RaterConfig):
    """Rater classification Bedrock Claude Config Class.
    The configuration primarily focuses on setting up the parameters for utilizing Bedrock to evaluate the
    correctness of answers in relation to given questions and contexts.

    Attributes:
        flow_name (str): Name of the rating flow, default is "RaterFlow".
        model_config (ModelConfig): Configuration for the Bedrock model. Includes aws_region ("us-west-2"), aws_profile ("default"),
        aws_access_key_id, aws_secret_key_id, aws_secret_access_key, aws_session_token, batch_size(1),
        model name ("anthropic.claude-v2"), batch_size (1), the server ("BedrockModelServer"), and the model_kwargs.
        label2score (Dict[str, float]): Mapping of labels to scores, default is {"Yes": 1.0, "No": 0.0}.
        prompt_template (PromptTemplate): Template for prompts used in rating. Includes instructions
                                               for rating, along with examples that detail the context, question,
                                               answer, label, and explanation for each case.
    """

    flow_name: str = "RaterFlow"
    model_config: ModelConfig = field(default_factory=BedrockModelConfig)
    label2score: Dict[str, float] = field(
        default_factory=lambda: {"Yes": 1.0, "No": 0.0}
    )
    prompt_template: PromptTemplate = field(
        default_factory=lambda: PromptTemplate(
            instruction="""Rate the answer based on the question and the context.
        Follow the format of the examples below to include context, question, answer, and label in the response.
        The response should not include examples in the prompt.""",
            few_shot_prompt=[
                Context(
                    context="""The Eiffel Tower, located in Paris, France, is one of the most famous landmarks in the
                    world. It was constructed in 1889 and stands at a height of 324 meters.""",
                    question="When was the Eiffel Tower constructed?",
                    answer="The Eiffel Tower was constructed in 1889.",
                    explanation="""The context explicitly mentions that the Eiffel Tower was constructed in 1889,
                    so the answer is correct.""",
                    label="Yes",
                ),
                Context(
                    context="""Photosynthesis is a process used by plants to convert light energy into chemical energy.
                    This process primarily occurs in the chloroplasts of plant cells.""",
                    question="Where does photosynthesis primarily occur in plant cells?",
                    answer="Photosynthesis primarily occurs in the mitochondria of plant cells.",
                    explanation="""The context mentions that photosynthesis primarily occurs in the chloroplasts of
                    plant cells, so the answer is incorrect.""",
                    label="No",
                ),
            ],
        )
    )


@dataclass
class RaterForGeneratedAnswerOpenAIGPT4Config(RaterConfig):
    """
    The configuration primarily focuses on setting up the parameters for utilizing GPT-4 to compare the
    correctness of generated answers with grounding answers in relation to given questions and contexts.

    Attributes:
        flow_name (str): Name of the rating flow, default is "RaterFlow".
        model_config (ModelConfig): Configuration for the GPT-4 model. Includes model name ("gpt-4"),
                                    the server ("OpenAIModelServer"), number of calls (1), temperature (0),
                                    and the response format (plain text).
        label2score (Dict[str, float]): Mapping of labels to scores, default is {"accept": 1.0,
                                        "equivalent": 0.0, "reject": -1.0}.
        prompt_template (PromptTemplate): Template for guided prompts used in rating. Includes instructions
                                               for rating, along with examples that detail the context, question,
                                               grounding answer, generated answer, label, and explanation for each case.
    """

    model_config: ModelConfig = field(
        default_factory=lambda: OpenAIModelConfig(
            model_name="gpt-4",
            model_server="OpenAIModelServer",
            num_call=1,
            temperature=0,
            response_format={"type": "text"},
        )
    )
    label2score: Dict[str, float] = field(
        default_factory=lambda: {
            "accept": 1.0,
            "equivalent": 0.0,
            "reject": -1.0,
        }
    )
    prompt_template: PromptTemplate = field(
        default_factory=lambda: PromptTemplate(
            instruction="""
            Compare two answers: a generated answer and a grounding answer based on a provided context and question.
            There are few annotated examples below, consisting of context, question, grounding answer, generated answer, explanation and label.
            If generated answer is better, you should give a higher score and vise versa. Check label to score dictionary: {label2score}.
            Your response should only focus on the unlabeled sample, including two fields: explanation and label (one of {label_list}).
            """,
            few_shot_prompt=[
                Context(
                    context="Early computers were built to perform a series of single tasks, like a calculator. Basic operating system could automatically run different programs in succession to speed up processing.",
                    question="Did early computers function like modern calculators?",
                    grounding_answer="No. Early computers were used primarily for complex calculating.",
                    generated_answer="Yes. Early computers were built to perform a series of single tasks, similar to a calculator.",
                    explanation="The generated answer is better because it correctly captures the essence of the early computers' functionality, which was to perform single tasks akin to calculators.",
                    label="accept",
                ),
                Context(
                    context="Operating systems did not exist in their modern and more complex forms until the early 1960s. Hardware features were added, that enabled use of runtime libraries, interrupts, and parallel processing.",
                    question="When did operating systems start to resemble their modern forms?",
                    grounding_answer="Operating systems started to resemble their modern forms in the early 1960s.",
                    generated_answer="Modern and more complex forms of operating systems began to emerge in the early 1960s.",
                    explanation="Both answers are equally good as they accurately pinpoint the early 1960s as the period when modern operating systems began to develop.",
                    label="equivalent",
                ),
                Context(
                    context="Operating systems did not exist in their modern and more complex forms until the early 1960s. Hardware features were added, that enabled use of runtime libraries, interrupts, and parallel processing.",
                    question="What features were added to hardware in the 1960s?",
                    grounding_answer="Hardware in the 1960s saw the addition of features like runtime libraries and parallel processing.",
                    generated_answer="The 1960s saw the addition of input output control and compatible timesharing capabilities in hardware.",
                    explanation="The generated answer is worse because it inaccurately suggests the addition of capabilities of hardware in 1960s which is not supported by the context.",
                    label="reject",
                ),
            ],
        )
    )


@dataclass
class RaterForGeneratedAnswerOpenAIGPT3p5Config(RaterConfig):
    """
    The configuration primarily focuses on setting up the parameters for utilizing GPT-3.5 to evaluate the
    correctness of answers in relation to given questions and contexts.

    Attributes:
        flow_name (str): Name of the rating flow, default is "RaterFlow".
        model_config (ModelConfig): Configuration for the GPT-3.5 model. Includes model name ("gpt-3-turbo-1106"),
                                    the server ("OpenAIModelServer"), number of calls (1), temperature (0),
                                    and the response format (plain text).
        label2score (Dict[str, float]): Mapping of labels to scores, default is {
                                        "accept": 1.0, "equivalent": 0.0, "reject": -1.0}.
        prompt_template (PromptTemplate): Template for guided prompts used in rating. Includes instructions
                                               for rating, along with examples that detail the context, question,
                                               grounding answer, generated answer, label, and explanation for each case.
    """

    model_config: ModelConfig = field(
        default_factory=lambda: OpenAIModelConfig(
            model_name="gpt-3.5-turbo-1106",
            model_server="OpenAIModelServer",
            num_call=1,
            temperature=0,
            response_format={"type": "text"},
        )
    )
    label2score: Dict[str, float] = field(
        default_factory=lambda: {
            "accept": 1.0,
            "equivalent": 0.0,
            "reject": -1.0,
        }
    )
    prompt_template: PromptTemplate = field(
        default_factory=lambda: PromptTemplate(
            instruction="""
            # Task: Evaluate and compare two answers: a generated answer and a grounding answer based on a provided context and question.
            ## Input: A sample to be labeled:
            1. context: A brief text containing key information.
            2. question: A query related to the context, testing knowledge that can be inferred or directly obtained from it.
            3. grounding Answer: Pre-formulated, usually from human.
            4. generated Answer: From a language model.
            ## Evaluation Criteria: If generated answer is better, you should give a higher score and vise versa. Check label to score dictionary: {label2score}.
            ## Response Format: Your response should only include two fields below:
            1. explanatoin: Reasoning behind your judgment, detailing why the generated answer is better, equivalent or worse.
            2. label: Your judgment (one of {label_list}).
            ## Note:
            Only use the example below as a few shot demonstrate but not include them in the final response. Your response should only focus on the unlabeled sample.
            """,
            few_shot_prompt=[
                Context(
                    context="Early computers were built to perform a series of single tasks, like a calculator. Basic operating system could automatically run different programs in succession to speed up processing.",
                    question="Did early computers function like modern calculators?",
                    grounding_answer="No. Early computers were used primarily for complex calculating.",
                    generated_answer="Yes. Early computers were built to perform a series of single tasks, similar to a calculator.",
                    explanation="The generated answer is better because it correctly captures the essence of the early computers' functionality, which was to perform single tasks akin to calculators.",
                    label="accept",
                ),
                Context(
                    context="Operating systems did not exist in their modern and more complex forms until the early 1960s. Hardware features were added, that enabled use of runtime libraries, interrupts, and parallel processing.",
                    question="When did operating systems start to resemble their modern forms?",
                    grounding_answer="Operating systems started to resemble their modern forms in the early 1960s.",
                    generated_answer="Modern and more complex forms of operating systems began to emerge in the early 1960s.",
                    explanation="Both answers are equally good as they accurately pinpoint the early 1960s as the period when modern operating systems began to develop.",
                    label="equivalent",
                ),
                Context(
                    context="Operating systems did not exist in their modern and more complex forms until the early 1960s. Hardware features were added, that enabled use of runtime libraries, interrupts, and parallel processing.",
                    question="What features were added to hardware in the 1960s?",
                    grounding_answer="Hardware in the 1960s saw the addition of features like runtime libraries and parallel processing.",
                    generated_answer="The 1960s saw the addition of input output control and compatible timesharing capabilities in hardware.",
                    explanation="The generated answer is worse because it inaccurately suggests the addition of capabilities of hardware in 1960s which is not supported by the context.",
                    label="reject",
                ),
            ],
        )
    )


###########################################################
#                    Pipeline Config                      #
###########################################################
@dataclass
class PipelineConfig:
    """MultiFlowsPipeline Config Class."""

    extract_config: ExtractConfig = field(default_factory=ExtractConfig)
    transform_config: TransformConfig = field(default_factory=TransformConfig)
