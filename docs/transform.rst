#########
TransformFlow
#########

With **uniflow** you can transform your split content into structured data.

For example, if you have a list of paragraphs, you can use **uniflow** to transform it into a list of structured questions and answers.

Example
=================

Here is some example code to give you an full overview. We'll break it down in subsequent sections.

.. code:: python

    from uniflow.flow.client import TransformClient
    from uniflow.flow.config import TransformOpenAIConfig
    from uniflow.op.model.model_config import OpenAIModelConfig
    from uniflow.op.prompt import PromptTemplate, Context

    raw_context_input = [
        "We report on a series of experiments with convolutional neural networks (CNN) trained on top of pre-trained word vectors for sentence-level classification tasks. We show that a simple CNN with little hyperparameter tuning and static vectors achieves excellent results on multiple benchmarks. Learning task-specific vectors through fine-tuning offers further gains in performance. We additionally propose a simple modification to the architecture to allow for the use of both task-specific and static vectors. The CNN models discussed herein improve upon the state of the art on 4 out of 7 tasks, which include sentiment analysis and question classification.",
        "Convolutional neural networks (CNN) utilize layers with convolving filters that are applied to local features [1]. Originally invented for computer vision, CNN models have subsequently been shown to be effective for NLP and have achieved excellent results in semantic parsing [13], search query retrieval [2], sentence modeling [1], and other traditional NLP tasks [1].	",
    ]

    guided_prompt = PromptTemplate(
        instruction="""Generate one question and its corresponding answer based on the last context in the last
        example. Follow the format of the examples below to include context, question, and answer in the response""",
        few_shot_prompt=[Context(
            context="In 1948, Claude E. Shannon published A Mathematical Theory of\nCommunication (Shannon, 1948) establishing the theory of\ninformation. In his article, Shannon introduced the concept of\ninformation entropy for the first time. We will begin our journey here.""",
            question="Who published A Mathematical Theory of Communication in 1948?""",
            answer="Claude E. Shannon."""
        )]
    )
    input_data = [
            Context(
                context=p,
                question="",
                answer="",
            )
            for p in raw_context_input
    ]

    config = TransformOpenAIConfig(
        prompt_template=guided_prompt,
        model_config=OpenAIModelConfig(
            response_format={"type": "json_object"}
        ),
    )
    transform_client = TransformClient(config)

    output = transform_client.run(input_data)


As you can see, you'll need to convert your input data into :code:`Context` objects, write a :code:`PromptTemplate` to guide the LLM, and then run your data with a :code:`TransformConfig` through a :code:`TransformClient`.

This will return a list of structured questions and answers by default:

.. code:: python

    [{'output': [{'response': ['context': 'We report on a series of experiments with convolutional neural networks (CNN) trained on top of pre-trained word vectors for sentence-level classification tasks. We show that a simple CNN with little hyperparameter tuning and static vectors achieves excellent results on multiple benchmarks. Learning task-specific vectors through fine-tuning offers further gains in performance. We additionally propose a simple modification to the architecture to allow for the use of both task-specific and static vectors. The CNN models discussed herein improve upon the state of the art on 4 out of 7 tasks, which include sentiment analysis and question classification.', 'question': 'What type of neural network was used in the experiments?', 'answer': 'Convolutional neural networks (CNN).'],
        'error': 'No errors.'}],
    'root': <uniflow.node.Node at 0x1169cae00>},
    {'output': [{'response': ['context': 'Convolutional neural networks (CNN) utilize layers with convolving filters that are applied to local features [1]. Originally invented for computer vision, CNN models have subsequently been shown to be effective for NLP and have achieved excellent results in semantic parsing [13], search query retrieval [2], sentence modeling [1], and other traditional NLP tasks [1].	', 'question': 'What are some tasks for which CNN models have been shown to be effective?', 'answer': 'CNN models have been shown to be effective for semantic parsing, search query retrieval, sentence modeling, and other traditional NLP tasks.'],
        'error': 'No errors.'}],
    'root': <uniflow.node.Node at 0x1169caa40>}]

Summary
=======
Use **uniflow**'s Transform flows to create structured data from unstructured data. The :code:`PromptTemplate` object guides the user in creating the prompt for the LLM to generate the structured data. The :code:`Context` object contains the context from which the LLM will create the structured data, as well as the structure for the LLM to follow.

Once the user has created the :code:`PromptTemplate` and :code:`Context` objects, they can run the :code:`TransformClient` to generate the structured data.

Finally, with this structured data, the user can further fine-tune their models.

.. toctree::
   :maxdepth: 4

   transform_client
   transform_config
   transform_prompt_template
