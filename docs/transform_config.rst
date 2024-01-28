TransformConfig
###############

The :code:`TransformConfig` is the configuration for the transform flow. It contains the following fields:

+-------------------+------------------+--------------------------------------------------+
| Field             | Type             | Description                                      |
+===================+==================+==================================================+
| model_config      | ModelConfig      | Configuration for the LLM model                  |
+-------------------+------------------+--------------------------------------------------+
| num_thread        | int              | Number of threads. Default is 1                  |
+-------------------+------------------+--------------------------------------------------+
| prompt_template   | PromptTemplate   | Template for the prompting the LLM. Contains an  |
|                   |                  | instruction and few shot examples for generating |
|                   |                  | questions and answers                            |
+-------------------+------------------+--------------------------------------------------+

This is the base configuration for the transform flow. We've also created a few pre-defined configurations for you to use.

Pre-defined Configurations
==========================
**uniflow** comes with several pre-defined configurations for you to use. You can find them in :code:`uniflow.flow.config`.

+------------------------------------------+--------------------------------------------+-----------------+----------------------------------------------------+
| Configuration                            | LLM                                        | Response format | Description                                        |
+==========================================+============================================+=================+====================================================+
| TransformOpenAIConfig                    | :code:`gpt-3.5-turbo-1106`                 | text            | Configuration for generating questions and answers |
|                                          |                                            |                 | using OpenAI's GPT-3.5 model                       |
+------------------------------------------+--------------------------------------------+-----------------+----------------------------------------------------+
| TransformHuggingFaceConfig               | :code:`mistralai/Mistral-7B-Instruct-v0.2` | text            | Base Configuration for generating questions and    |
|                                          |                                            |                 | answers using HuggingFace's Mistral-7B model       |
+------------------------------------------+--------------------------------------------+-----------------+----------------------------------------------------+
| TransformQAHuggingFaceConfig             | :code:`mistralai/Mistral-7B-Instruct-v0.2` | text            | Configuration for generating questions and answers |
|                                          |                                            |                 | using HuggingFace's Mistral-7B model               |
+------------------------------------------+--------------------------------------------+-----------------+----------------------------------------------------+
| TransformQAHuggingFaceJsonFormatConfig   | :code:`mistralai/Mistral-7B-Instruct-v0.2` | json            | Configuration for generating questions and answers |
|                                          |                                            |                 | using HuggingFace's Mistral-7B model               |
+------------------------------------------+--------------------------------------------+-----------------+----------------------------------------------------+
| TransformLMQGConfig                      | :code:`lmqg/t5-base-squad-qg-ae`           |  tuple          | Configuration for generating questions and answers |
|                                          |                                            |                 | using LMQG                                         |
+------------------------------------------+--------------------------------------------+-----------------+----------------------------------------------------+
| TransformForGenerationOpenAIGPT3p5Config | :code:`gpt-3.5-turbo-1106`                 | text            | Configuration for generating questions and answers |
|                                          |                                            |                 | using OpenAI's GPT-3.5 model.                      |
+------------------------------------------+--------------------------------------------+-----------------+----------------------------------------------------+
| TransformForGenerationOpenAIGPT3p5Config | :code:`gpt-3.5-turbo-1106`                 | json            | Configuration for generating a list of outputs     |
|                                          |                                            |                 | using OpenAI's GPT-3.5 model.                      |
+------------------------------------------+--------------------------------------------+-----------------+----------------------------------------------------+
| TransformForClusteringOpenAIGPT4Config   | :code:`gpt-4`                              | json            | Configuration for clustering the contexts into     |
|                                          |                                            |                 | groups using OpenAI's GPT-4 model.                 |
+------------------------------------------+--------------------------------------------+-----------------+----------------------------------------------------+

ModelConfig
===========
For more details on these :code:`ModelConfigs` and their default parameters, please see the |model_config_link|.

.. |model_config_link| raw:: html

   <a href="https://github.com/CambioML/uniflow/blob/main/uniflow/op/model/model_config.py" target="_blank" rel="noopener noreferrer">model_config module</a>

....

Next, we'll dive into writing prompts for **uniflow** using the :code:`PromptTemplates`.