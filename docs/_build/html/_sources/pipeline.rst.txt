MultiFlowsPipeline
########

With the Pipeline, you can link several different **uniflow** flows to create a single flow.

Example
=======
In this example, we'll create a pipeline to do that following:
  #. Extract text from a PDF file
  #. Generate questions and answers for the text

To do this, we'll have to link up our :code:`ExtractPDFFlow` and :code:`TransformFlow` flows into a :code:`MultiFlowsPipeline`.

.. code-block:: python

    from uniflow.pipeline import MultiFlowsPipeline
    from uniflow.flow.config import PipelineConfig
    from uniflow.flow.config import TransformOpenAIConfig, ExtractPDFConfig
    from uniflow.flow.config import OpenAIModelConfig, NougatModelConfig
    from uniflow.op.prompt import PromptTemplate, Context
    from uniflow.op.extract.split.constants import PARAGRAPH_SPLITTER

    data = [
        {"pdf": input_file_path},
    ]

    # 1. Configure our PDF extract flow

    extract_config = ExtractPDFConfig(
        model_config=NougatModelConfig(
            model_name = "0.1.0-small",
            batch_size = 1 # When batch_size>1, nougat will run on CUDA, otherwise it will run on CPU
        ),
        splitter="fads",
    )

    # 2. Configure our QA generation flow
    guided_prompt = PromptTemplate(
        instruction="""Generate one question and its corresponding answer based on the last context in the last
        example. Follow the format of the examples below to include context, question, and answer in the response""",
        few_shot_prompt=[Context(
            context="In 1948, Claude E. Shannon published A Mathematical Theory of\nCommunication (Shannon, 1948) establishing the theory of\ninformation. In his article, Shannon introduced the concept of\ninformation entropy for the first time. We will begin our journey here.""",
            question="Who published A Mathematical Theory of Communication in 1948?""",
            answer="Claude E. Shannon."""
        )]
    )

    transform_config = TransformOpenAIConfig(
        prompt_template=guided_prompt,
        model_config=OpenAIModelConfig(
            response_format={"type": "json_object"}
        ),
    )

    # 3. Configure our pipeline
    p = MultiFlowsPipeline(PipelineConfig(
        extract_config=extract_config,
        transform_config=transform_config,
    ))

    # 4. Run the pipeline
    output = p.run(data)

As you can see, we first set up our :code:`ExtractPDFConfig` and :code:`TransformOpenAIConfig` configs, just as we normally do. Then, we pass them into our :code:`PipelineConfig` object which, we pass that into our :code:`MultiFlowsPipeline` object. This allows us to seamlessly run our extract and transform flows in a single pipeline.

After running the pipeline, we get the following output:

.. code-block:: python

    [[{'output': [{'response': [{'context': 'In 1948, Claude E. Shannon published A Mathematical Theory of\nCommunication (Shannon, 1948) establishing the theory of\ninformation. In his article, Shannon introduced the concept of\ninformation entropy for the first time. We will begin our journey here.',
        'question': 'What concept did Claude E. Shannon introduce for the first time in his article A Mathematical Theory of Communication?',
        'answer': 'Claude E. Shannon introduced the concept of information entropy for the first time.'}],
        'error': 'No errors.'}],
        'root': <uniflow.node.Node at 0x17a8b7550>},
        {'output': [{'response': [{'context': 'We report on a series of experiments with convolutional neural networks (CNN) trained on top of pre-trained word vectors for sentence-level classification tasks. We show that a simple CNN with little hyperparameter tuning and static vectors achieves excellent results on multiple benchmarks. Learning task-specific vectors through fine-tuning offers further gains in performance. We additionally propose a simple modification to the architecture to allow for the use of both task-specific and static vectors. The CNN models discussed herein improve upon the state of the art on 4 out of 7 tasks, which include sentiment analysis and question classification.',
            'question': 'What type of neural networks were used in the experiments reported in the context?',
            'answer': 'Convolutional neural networks (CNN) were used in the experiments.'}],
            'error': 'No errors.'}],
        'root': <uniflow.node.Node at 0x17a8b7010>},
            ...
    ]]

For the full example and breakdown, see |notebook_link|.

.. |notebook_link| raw:: html

   <a href="https://github.com/CambioML/uniflow/blob/main/example/pipeline/pipeline_pdf.ipynb" target="_blank" rel="noopener noreferrer">this notebook</a>



....

Next, we'll see how you can use **uniflow's** :code:`Rater` to rate different data.