Installation
===================================

**uniflow** is an open-source data curation platform for LLMs. Using **uniflow**,
everyone can create structured data from unstructured data.


Quick Start
-----------
Getting started is easy, simply :code:`pip install` the **uniflow** library:

.. code:: bash

  pip3 install uniflow

In-depth Installation
---------------------
To get started with **uniflow**, you can install it using :code:`pip` in a conda environment.

First, create a conda environment on your terminal using:

.. code:: bash

  conda create -n uniflow python=3.10 -y
  conda activate uniflow  # some OS requires `source activate uniflow`

Next, install the compatible pytorch based on your OS.

If you are on a GPU, install pytorch based on your cuda version. You can find your CUDA version via nvcc -V.

.. code:: bash

  pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121  # cu121 means cuda 12.1

If you are on a CPU instance,

.. code:: bash

  pip3 install torch

Then, install uniflow:

.. code:: bash

  pip3 install uniflow

If you are running the :code:`HuggingfaceModelFlow`, you will also need to install the :code:`transformers`, :code:`accelerate`, :code:`bitsandbytes`, :code:`scipy` libraries:

.. code:: bash

  pip3 install transformers accelerate bitsandbytes scipy

Finally, if you are running the :code:`LMQGModelFlow`, you will also need to install the :code:`lmqg` and :code:`spacy` libraries:

.. code:: bash

  pip3 install lmqg spacy

Congrats you have finished the installation!

.. note::

   This project is under active development!
