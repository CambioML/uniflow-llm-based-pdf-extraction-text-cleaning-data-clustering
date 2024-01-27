ExtractFlow
===================================

With **uniflow** you can extract and split from unstructured text including
    - PDFs
    - HTML
    - Images
    - Markdown
    - Slides
    - Tables

Here is some example code to get you started:

.. code:: python

  from uniflow.flow.client import ExtractClient
  from uniflow.flow.config import ExtractPDFConfig
  from uniflow.op.model.model_config import NougatModelConfig
  from uniflow.op.extract.split.constants import PARAGRAPH_SPLITTER

  data = [
    {"filename": input_file_path},
  ]

  config = ExtractPDFConfig(
    model_config=NougatModelConfig(
      model_name = "0.1.0-small",
      batch_size = 1 # When batch_size>1, nougat will run on CUDA, otherwise it will run on CPU
    ),
    splitter=PARAGRAPH_SPLITTER,
  )
  nougat_client = ExtractClient(config)

  output = nougat_client.run(data)

This will take the input file located at **input_file_path**, extract the text using the Nougat Model, and split it into paragraphs. The output will contain a dictionary with a `text` key for each file, which contains a list of the extracted paragraphs.

.. code:: python

    [{'output': [{'text': ['# Convolutional Neural Networks for Sentence Classification',
        ' Yoon Kim',
        'New York University',
        'yhk255@nyu.edu',
        '###### Abstract',
        'We report on a series of experiments with convolutional neural networks (CNN) trained on top of pre-trained word vectors for sentence-level classification tasks. We show that a simple CNN with little hyperparameter tuning and static vectors achieves excellent results on multiple benchmarks. Learning task-specific vectors through fine-tuning offers further gains in performance. We additionally propose a simple modification to the architecture to allow for the use of both task-specific and static vectors. The CNN models discussed herein improve upon the state of the art on 4 out of 7 tasks, which include sentiment analysis and question classification.',
        ...]}]
    }]

With this split text, you can further use **uniflow** to transform the text into structured data, such as questions and answers.

For a more in-depth example, you can check out |notebook_link|.

.. |notebook_link| raw:: html

   <a href="https://github.com/CambioML/uniflow/tree/main/example/extract" target="_blank" rel="noopener noreferrer">these notebooks</a>

.. toctree::
   :maxdepth: 4

   extract_client
   extract_config