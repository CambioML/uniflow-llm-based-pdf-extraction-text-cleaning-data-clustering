"""Op module for uniflow."""
import abc
from typing import Sequence

import uniflow.op.utils as utils
from uniflow.node import Node


class OpScope:
    """Op scope class for uniflow."""

    def __init__(self, name: str) -> None:
        """Op scope class constructor for uniflow.

        Args:
            name (str): Name of the op scope.
        """
        self._name = name

    def __enter__(self) -> None:
        """Enter op scope."""
        utils.OPS_SCOPE.append(self._name)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit op scope."""
        utils.OPS_SCOPE.pop()


class Op(abc.ABC):
    """Op class for uniflow."""

    def __init__(self, name: str) -> None:
        """Constructor of op class for uniflow.

        Args:
            name (str): Name of the op.
        """
        self._scope_name = utils.get_op_scope_name(name)
        self._count = 0

    def clear_count(self) -> None:
        """Clear count of the op."""
        self._count = 0

    def _increase_count(self) -> None:
        """Increase count of the op."""
        self._count += 1

    def name(self) -> str:
        """Name of the op."""
        return self._scope_name

    def unique_name(self) -> str:
        """Unique name of the op."""
        self._increase_count()
        return f"{self._scope_name}_{self._count}"

    @abc.abstractmethod
    def __call__(self, *args: Sequence[Node]) -> Sequence[Node]:
        """Call op."""
        raise NotImplementedError("Not implemented yet.")
