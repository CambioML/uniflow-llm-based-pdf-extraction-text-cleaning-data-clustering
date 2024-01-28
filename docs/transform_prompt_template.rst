PromptTemplate
#########
In order to transform our data using LLMs, we'll need to prompt the LLM to generate the data we want. This is done by creating a :code:`PromptTemplate` object which we'll pass to our :code:`TransformClient` via the :code:`TransformConfig`.

The :code:`PromptTemplate` object contains the following fields:

+------------------------+------------+--------------------------------------------------+
| Field                  | Type       | Description                                      |
+========================+============+==================================================+
| instruction            | string     | A string to guide the LLM in generating the data |
+------------------------+------------+--------------------------------------------------+
| few_shot_template      | [Context]  | List of few-shot examples to guide the LLM       |
+------------------------+------------+--------------------------------------------------+

Here's an example of how to use the :code:`PromptTemplate` object:

.. code:: python

    from uniflow.flow.client import TransformClient
    from uniflow.flow.config import TransformOpenAIConfig
    from uniflow.op.prompt import PromptTemplate, Context


    guided_prompt = PromptTemplate(
        instruction="""Generate one question and its corresponding answer based on the last context in the last
        example. Follow the format of the examples below to include context, question, and answer in the response""",
        few_shot_prompt=[Context(
            context="In 1948, Claude E. Shannon published A Mathematical Theory of\nCommunication (Shannon, 1948) establishing the theory of\ninformation. In his article, Shannon introduced the concept of\ninformation entropy for the first time. We will begin our journey here.""",
            question="Who published A Mathematical Theory of Communication in 1948?""",
            answer="Claude E. Shannon."""
        )]
    )

    config = TransformOpenAIConfig(prompt_template=guided_prompt)

    transform_client = TransformClient(config)

    output = transform_client.run(input_data)

As you can see, we are instructing the LLM to generate a question and answer, and also to follow our :code:`few_shot_prompt`. The :code:`few_shot_prompt` contains a single :code:`Context` object with an example :code:`context`, :code:`question`, and :code:`answer`. The :code:`PromptTemplate` is then passed to the :code:`TransformConfig` object which is passed to the :code:`TransformClient` object. The :code:`TransformClient` object is then used to run the :code:`input_data` through the LLM.

....

Next, we'll see how we can tie our :code:`ExtractFlow` and :code:`TransformFlow` together to create a :code:`MultiFlowsPipeline`.