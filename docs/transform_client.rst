TransformClient
===============
The :code:`TransformClient` is the main entry point for the transform flow. It takes in a :code:`TransformConfig` and runs the data through the flow.

.. code:: python

    from uniflow.flow.client import TransformClient
    from uniflow.flow.config import TransformOpenAIConfig

    transform_client = TransformClient(TransformOpenAIConfig())

    output = transform_client.run(input_data)

....

Next, we'll dig into the :code:`TransformConfig`.