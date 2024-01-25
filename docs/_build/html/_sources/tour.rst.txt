Quick Tour
===================================

Using **uniflow**, one may easily extract, transform, and rate data harnessing the power of LLMs from You can utilize LLMs from OpenAI, Hugging Face, Anthropic, and more.

:ref:`ExtractFlow`
-----------------------------------
Extract and split from unstructured text including:
    - PDFs
    - TXT
    - Images
    - Markdown
    - Jupyter Notebooks

.. code-block:: python

    from uniflow.flow.client import ExtractClient
    from uniflow.flow.config import ExtractPDFConfig
    from uniflow.op.model.model_config import NougatModelConfig

    config = ExtractPDFConfig(
        model_config=NougatModelConfig()
    )

    nougat_client = ExtractClient(config)

    output = nougat_client.run(input_data)

:ref:`TransformFlow`
-----------------------------------
Transform content into structured data, e.g. create questions and answers from text.

.. code-block:: python

    from uniflow.flow.client import TransformClient
    from uniflow.flow.config import TransformOpenAIConfig
    from uniflow.op.model.model_config import OpenAIModelConfig

    config = TransformOpenAIConfig(
        prompt_template=guided_prompt,
        model_config=OpenAIModelConfig(),
    )
    transform_client = TransformClient(config)

    output = transform_client.run(input_data)

:ref:`MultiFlowsPipeline`
-----------------------------------
With the Pipeline, you can link several different **uniflow** flows to create a single flow.

.. code-block:: python

    from uniflow.pipeline import MultiFlowsPipeline
    from uniflow.flow.config import PipelineConfig
    from uniflow.flow.config import TransformOpenAIConfig, ExtractPDFConfig
    from uniflow.flow.config import OpenAIModelConfig, NougatModelConfig

    data = [
        {"pdf": input_file_path},
    ]

    # 1. Configure our PDF extract flow

    extract_config = ExtractPDFConfig(
        model_config=NougatModelConfig(),
    )

    # 2. Configure our transform flow
    transform_config = TransformOpenAIConfig(
        prompt_template=guided_prompt,
        model_config=OpenAIModelConfig(),
    )

    # 3. Configure our pipeline
    p = MultiFlowsPipeline(PipelineConfig(
        extract_config=extract_config,
        transform_config=transform_config,
    ))

    # 4. Run the pipeline
    output = p.run(data)

:ref:`Rater`
-----------------------------------
Rate and compare the quality of the data

.. code:: python

   from uniflow.flow.client import RaterClient
   from uniflow.flow.config  import (
      RaterForClassificationHuggingfaceConfig,
      HuggingfaceModelConfig,
   )

   config = RaterForClassificationHuggingfaceConfig(
      model_config=HuggingfaceModelConfig()
   )

   client = RaterClient(config)
   output = client.run(input_data)

....

Our goal is to close the gap between active learning and data collection. We make
it easy to produce structured data for your own models, so you can use that data to finetune your models.

Next, we'll dive into the :code:`Context` object.
