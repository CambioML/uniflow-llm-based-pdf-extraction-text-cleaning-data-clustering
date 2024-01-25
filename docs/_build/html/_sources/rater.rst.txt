Rater
#####

With **uniflow** you can use the power of LLMs to automatically validate the quality and compliance of your models.

Example
=================

Here is some example code to give you an full overview. We'll break it down in subsequent sections.

.. code:: python

   from uniflow.flow.client import RaterClient
   from uniflow.flow.config  import (
      RaterForClassificationHuggingfaceConfig,
      HuggingfaceModelConfig,
   )
   from uniflow.op.prompt import Context

   raw_input = [
      ("The Pacific Ocean is the largest and deepest of Earth's oceanic divisions. It extends from the Arctic Ocean in the north to the Southern Ocean in the south.",
      "What is the largest ocean on Earth?",
      "The largest ocean on Earth is the Pacific Ocean."), # correct
      ("Shakespeare, a renowned English playwright and poet, wrote 39 plays during his lifetime. His works include famous plays like 'Hamlet' and 'Romeo and Juliet'.",
      "How many plays did Shakespeare write?",
      "Shakespeare wrote 31 plays."), # incorrect
      ("The human brain is an intricate organ responsible for intelligence, memory, and emotions. It is made up of approximately 86 billion neurons.",
      "What is the human brain responsible for?",
      "The human brain is responsible for physical movement."), # incorrect
   ]

   data = [
      Context(context=c[0], question=c[1], answer=c[2])
      for c in raw_input
   ]
   config = RaterForClassificationHuggingfaceConfig(
      model_config=HuggingfaceModelConfig(
         response_start_key="explanation",
         response_format={"type": "json_object"},
         batch_size=2
      )
   )

   client = RaterClient(config)
   output = client.run(data)

As you can see, you'll need to convert your input data into :code:`Context` objects, set up your **RaterClassificationConfig**, and run your data through a :code:`RaterClient`.

This will return the following output:

.. code:: python

   [{'output': [{'error': 'No errors.',
              'response': [{'average_score': 1.0,
                            'majority_vote': 'yes',
                            'samples': [{'answer': 'The largest ocean on Earth '
                                                   'is the Pacific Ocean.',
                                         'context': 'The Pacific Ocean is the '
                                                    'largest and deepest of '
                                                    "Earth's oceanic "
                                                    'divisions. It extends '
                                                    'from the Arctic Ocean in '
                                                    'the north to the Southern '
                                                    'Ocean in the south.',
                                         'explanation': 'The answer is '
                                                        'consistent with the '
                                                        'fact stated in the '
                                                        'context that the '
                                                        'Pacific Ocean is the '
                                                        'largest ocean on '
                                                        'Earth, so the answer '
                                                        'is correct.',
                                         'label': 'Yes.',
                                         'question': 'What is the largest '
                                                     'ocean on Earth?'}],
                            'scores': [1.0],
                            'votes': ['yes']},
                           {'average_score': 0.0,
                            'majority_vote': 'no',
                            'samples': [{'answer': 'Shakespeare wrote 31 '
                                                   'plays.',
                                         'context': 'Shakespeare, a renowned '
                                                    'English playwright and '
                                                    'poet, wrote 39 plays '
                                                    'during his lifetime. His '
                                                    'works include famous '
                                                    "plays like 'Hamlet' and "
                                                    "'Romeo and Juliet'.",
                                         'explanation': 'The answer is '
                                                        'inconsistent with the '
                                                        'fact stated in the '
                                                        'context that '
                                                        'Shakespeare wrote 39 '
                                                        'plays, therefore the '
                                                        'answer is incorrect.',
                                         'label': 'No',
                                         'question': 'How many plays did '
                                                     'Shakespeare write?'}],
                            'scores': [0.0],
                            'votes': ['no']}]}],
  'root': <uniflow.node.Node object at 0x7f67989c2080>},
  ...]

As you can see, the :code:`Rater` will return the average score and majority vote for each sample, as well as the individual scores and votes for each sample. It will also return the error message, if any, for each sample.

For the full example and breakdown, see |notebook_link|.

.. |notebook_link| raw:: html

   <a href="https://github.com/CambioML/uniflow/blob/main/example/rater/huggingface_classification.ipynb" target="_blank" rel="noopener noreferrer">this notebook</a>

....

Next, dive into the code to learn more about how to use **uniflow**.
