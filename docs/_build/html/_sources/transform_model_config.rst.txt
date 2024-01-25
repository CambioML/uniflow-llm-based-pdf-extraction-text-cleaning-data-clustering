ModelConfig
###########
In the :code:`TransformFlow`, you use the :code:`ModelConfig` to define the model you want to use for your task.

Here's a table of the different :code:`ModelConfigs` you can use:

+------------------------------------------+--------------------------------------------+-------------------------+
| Configuration                            | LLM                                        | Default Response Format |
+==========================================+============================================+=========================+
| OpenAIModelConfig                        | :code:`gpt-3.5-turbo-1106`                 | text                    |
+------------------------------------------+--------------------------------------------+-------------------------+
| AzureOpenAIModelConfig                   | :code:`gpt-3.5-turbo-1106`                 | text                    |
+------------------------------------------+--------------------------------------------+-------------------------+
| HuggingfaceModelConfig                   | :code:`mistralai/Mistral-7B-Instruct-v0.2` |                         |
+------------------------------------------+--------------------------------------------+-------------------------+
| LMQGModelConfig                          | :code:`mistralai/Mistral-7B-Instruct-v0.2` |                         |
+------------------------------------------+--------------------------------------------+-------------------------+

For more details on these :code:`ModelConfigs` and their default parameters, please see the `model_config module <https://github.com/CambioML/uniflow/blob/main/uniflow/op/model/model_config.py>`_

....

Next, we'll dive into writing prompts for **uniflow** using the :code:`PromptTemplates`.