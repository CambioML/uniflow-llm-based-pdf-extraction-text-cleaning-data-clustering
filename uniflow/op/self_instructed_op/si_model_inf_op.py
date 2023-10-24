"""Model inference operation."""
import copy
import logging

from typing import Any, Mapping
from uniflow.op.basic.linear_op import LinearOp
import openai

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling

import torch
from torch.utils.data import Dataset, random_split
from transformers import TrainingArguments, Trainer
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import uniflow.flow.constants as constants


class SIModelInfOp(LinearOp):
    """Model inference operation for Self-Instructed fine tuning.

    Args:
        nodes (Sequence[Node]): Input nodes.

    Returns:
        Sequence[Node]: Output nodes.
    """
    def __init__(self, name: str):
        """Initialize SIModelInfOp class."""
        super().__init__(name)
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

    def _transform(self, value_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Call the language model to generate outputs for the prompt.
        Args:
            value_dict (Mapping[str, Any]): Input value dict.

        Returns:
            Mapping[str, Any]: Output value dict.
        """
        self._logger.info("Starting SIModelInfOp...")

        pages = value_dict[constants.PAGES_KEY][:]

        BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

        device_map = "auto"

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token

        # initialize model
        self._logger.info("1. Initializing model...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map=device_map,
            offload_folder="./offload",
        )

        self._logger.info("2. Initializing pipeline...")
        # initialize pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            # use_cache=True,
            # do_sample=True,
            # temperature = 0.3,
            ##top_p = 0.92,
            # top_k=5,
            # max_length=1000,
            device_map="auto",
            max_new_tokens=768,
            num_return_sequences=1,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        # Create prompt template

        prompt_template_trn = """<s>[INST] Generate one question from each paragraph in the context, provide answer to each question in a few sentences according to the context. Present the question within one single line starting with 'Q: ' and the answer within one single line starting with "A: ". Add " == End == " at the end of each answer.

        {context} [/INST] </s>
        """
        PROMPT_trn = PromptTemplate(
            template=prompt_template_trn, input_variables=["context"]
        )

        # Create LangChain LLMChain
        self._logger.info("3. Creating LangChain LLMChain...")
        chain_trn = LLMChain(
            llm=llm,
            prompt=PROMPT_trn,
        )
        self._logger.info("1-3 Complete!")

        text_line_q = []
        text_line_in = []
        text_line_a = []

        self._logger.info("Starting to process pages...")
        for i in range(len(pages)):
            docs = pages[i].page_content
            self._logger.info(f"Processing page {i + 1} of {len(pages)}...")
            self._logger.debug(f"Training Content:\n {docs[:100]}...")
            response = chain_trn({"context": docs}, return_only_outputs=True)
            text = response["text"]
            self._logger.debug(f"Page {i + 1} \n {text} \n ========================== \n")
            for item in text.split("Q:"):
                self._logger.debug(f"Processing {item}\nLength {len(item)}")
                if len(item) > 0:
                    one_q_a = item.strip()
                    self._logger.debug(f"one_q_a = {one_q_a} ===")
                    if "A:" in one_q_a:
                        question = (
                            one_q_a.split("A:")[0].strip() + "[Page " + str(i) + "]"
                        )
                        self._logger.debug(f"Question: {question}")
                        text_line_q.append(question)

                        text_line_in.append("")

                        answer = one_q_a.split("A:")[1].strip()
                        self._logger.debug(f"Answer: {answer}")
                        text_line_a.append(answer)

            self._logger.info(f"=== processed page {i + 1} | total questions generated: {len(text_line_q)} ===")

        self._logger.info("Page processing complete!")
        self._logger.info("SIModelInfOp Complete!")

        return {
            "text_line_q": text_line_q,
            "text_line_in": text_line_in,
            "text_line_a": text_line_a,
        }
