Context
#######
The :code:`Context` object is used by **uniflow** to describe the input data. As such, we use it to wrap our input data in all our different flows. It's also used in our :code:`few_shot_prompt` examples for our :code:`TransformFlow` to help describe the desired output data structure.

The :code:`Context` object contains the following fields:

+--------------------------+---------+-----------------------------------------------------+
| Field                    | Type    | Description                                         |
+==========================+=========+=====================================================+
| **context**              | string  | the context from which the LLM will create the data |
+--------------------------+---------+-----------------------------------------------------+
| **additional fields**    | string  | additional fields, such as :code:`question` and     |
|                          |         | :code:`answer` to define the structure for the data |
+--------------------------+---------+-----------------------------------------------------+

The rest of the :code:`Context` is flexible to be created by the user. The user can create a :code:`Context` object with a question and answer, or a summary, etc.. The LLM will follow this Context to create the structured data output from all the input contexts.


Example
-----------------
For example, if you want to generate summaries from text in a :code:`TransformFlow`, you can use :code:`Context` as follows:

.. code:: python

    from uniflow.flow.client import TransformClient
    from uniflow.flow.config import TransformOpenAIConfig
    from uniflow.op.prompt import PromptTemplate, Context

    raw_context_input = [
        "We report on a series of experiments with convolutional neural networks (CNN) trained on top of pre-trained word vectors for sentence-level classification tasks. We show that a simple CNN with little hyperparameter tuning and static vectors achieves excellent results on multiple benchmarks. Learning task-specific vectors through fine-tuning offers further gains in performance. We additionally propose a simple modification to the architecture to allow for the use of both task-specific and static vectors. The CNN models discussed herein improve upon the state of the art on 4 out of 7 tasks, which include sentiment analysis and question classification.",
        "Convolutional neural networks (CNN) utilize layers with convolving filters that are applied to local features [1]. Originally invented for computer vision, CNN models have subsequently been shown to be effective for NLP and have achieved excellent results in semantic parsing [13], search query retrieval [2], sentence modeling [1], and other traditional NLP tasks [1].	",
    ]

    guided_prompt = PromptTemplate(
        instruction="Generate a one sentence summary based on the last context below. Follow the format of the examples below to include context and summary in the response",
        few_shot_prompt=[Context(
            context="When you're operating on the maker's schedule, meetings are a disaster. A single meeting can blow a whole afternoon, by breaking it into two pieces each too small to do anything hard in. Plus you have to remember to go to the meeting. That's no problem for someone on the manager's schedule. There's always something coming on the next hour; the only question is what. But when someone on the maker's schedule has a meeting, they have to think about it.",
            summary="Meetings disrupt the productivity of those following a maker's schedule, dividing their time in
        )]
    )
    input_data = [
            Context(
                context=c,
                summary="",
            )
            for c in raw_context_input
    ]
    config = TransformOpenAIConfig(prompt_template=guided_prompt)

    transform_client = TransformClient(config)

    output = transform_client.run(input_data)

    print(output[0]['output'][0]['response'])

    >>> {'context': 'We report on a series of experiments with convolutional neural networks (CNN) trained on top of pre-trained word vectors for sentence-level classification tasks. We show that a simple CNN with little hyperparameter tuning and static vectors achieves excellent results on multiple benchmarks. Learning task-specific vectors through fine-tuning offers further gains in performance. We additionally propose a simple modification to the architecture to allow for the use of both task-specific and static vectors. The CNN models discussed herein improve upon the state of the art on 4 out of 7 tasks, which include sentiment analysis and question classification.',
    'summary': 'A series of experiments with convolutional neural networks (CNN) trained on pre-trained word vectors for sentence-level classification tasks demonstrates that a simple CNN with little hyperparameter tuning and static vectors achieves excellent results on multiple benchmarks, and task-specific vectors through fine-tuning offer further gains in performance.',}

Note that both the :code:`context` and :code:`summary` fields are required in the :code:`Context` object for both the :code:`input_data` and the :code:`few_shot_prompt`. The :code:`summary` field is empty in the input data, but is filled in the :code:`few_shot_prompt` field of the :code:`PromptTemplate` object.

You can see further examples of how to use the :code:`Context` object in the :code:`ExtractFlow` and :code:`RateFlow` sections of the documentation.

....

Next, we'll learn about how you can use **uniflow** to extract and split unstructured data using the :code:`ExtractFlow`.