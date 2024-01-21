# Backend Interview README

## Part3
- ``value_dict`` type: type hint of [Node](../uniflow/node.py) shows that ``value_dict`` should be Mapping, but according to the code of batch and server_client result of ``TransformCopyFlow``, type of ``value_dict`` is actually a sequence of mapping. Nodes created by ``TransformCopyFlow`` are of type ``sequence of mapping``.
