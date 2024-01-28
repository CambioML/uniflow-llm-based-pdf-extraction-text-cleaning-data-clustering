ExtractClient
#####################
The :code:`ExtractClient` is the main entry point for the Extract flow. It takes in a :code:`ExtractConfig` and runs the data through the flow.

.. code:: python

    from uniflow.flow.client import ExtractClient
    from uniflow.flow.config import ExtractPDFConfig

    nougat_client = ExtractClient(ExtractPDFConfig())

    output = nougat_client.run(data)

....

Next, we'll dig into the :code:`ExtractConfig`.