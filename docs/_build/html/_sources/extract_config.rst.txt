ExtractConfig
#####################

The :code:`ExtractConfig` is the configuration for the Extract flow. It contains the following fields:

+--------------------------+------------------+-------------------------------------------------------+
| Field                    | Type             | Description                                           |
+==========================+==================+=======================================================+
| num_thread               | int              | Number of threads. Default is 1                       |
+--------------------------+------------------+-------------------------------------------------------+
| splitter (optional)      | string           | String pattern used to split the input file           |
+--------------------------+------------------+-------------------------------------------------------+
| model_config (optional)  | ModelConfig      | Configuration for the LLM model used for the extract  |
+--------------------------+------------------+-------------------------------------------------------+

This is the base configuration for the extract flow. We've also created a few pre-defined configurations for you to use.

Pre-defined Configurations
==========================
**uniflow** comes with several pre-defined configurations for you to use. You can find them in :code:`uniflow.flow.config`.

+------------------------------------------+-----------+-----------------+------------------------------+---------------------------------------------------+
| Configuration                            | File type | Splitter        | Model                        |Description                                        |
+==========================================+===========+=================+==============================+===================================================+
| ExtractTxtConfig                         | txt       | none            | none                         | Configuration for extracting content from .txt    |
+------------------------------------------+-----------+-----------------+------------------------------+---------------------------------------------------+
| ExtractPDFConfig                         | pdf       | paragraph       | Nougat                       | Configuration for extracting content from .pdf    |
|                                          |           |                 |                              | files.                                            |
+------------------------------------------+-----------+-----------------+------------------------------+---------------------------------------------------+
| ExtractImageConfig                       | image     | paragraph       | unstructuredio/yolo_x_layout | Configuration for extracting content from images  |
+------------------------------------------+-----------+-----------------+------------------------------+---------------------------------------------------+
| ExtractMarkdownConfig                    | markdown  | markdown header | none                         | Configuration for extracting content from markdown|
+------------------------------------------+-----------+-----------------+------------------------------+---------------------------------------------------+
| ExtractIpynbConfig                       | ipynb     | none            | none                         | Configuration for extracting content from Jupyter |
|                                          |           |                 |                              | Notebook (.ipynb) files.                          |
+------------------------------------------+-----------+-----------------+------------------------------+---------------------------------------------------+

....

Next, we'll see how we can transform our data using the :code:`TransformFlow`.