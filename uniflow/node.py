""" Node module for uniflow."""
from typing import Any, Mapping, Optional, Sequence


class Node:
    """Node class for uniflow."""

    def __init__(
        self,
        name: str,
        value_dict: Mapping[str, Any],
        prev_nodes: Optional[Sequence["Node"]] = None,
        next_nodes: Optional[Sequence["Node"]] = None,
        is_end: Optional[bool] = False,
    ) -> None:
        """Node class constructor for uniflow.

        Args:
            name (str): Name of the node.
            value_dict (Mapping[str, Any]): Value dictionary of the node.
            prev_nodes (Optional[Sequence["Node"]]): Previous nodes of the node.
            next_nodes (Optional[Sequence["Node"]]): Next nodes of the node.
            is_end (Optional[bool]): Whether the node is end node or not.
        """
        self._name = name
        self._value_dict = value_dict
        self._is_end = is_end
        self._prev_nodes = list(prev_nodes) if prev_nodes else []
        self._next_nodes = list(next_nodes) if next_nodes else []
        for node in self._prev_nodes:
            node.add_next_node(self)

    @property
    def name(self):
        """Name of the node."""
        return self._name

    @property
    def value_dict(self):
        """Value dictionary of the node."""
        return self._value_dict

    @property
    def is_end(self):
        """Whether the node is end node or not."""
        return self._is_end

    @property
    def prev_nodes(self):
        """Previous nodes of the node."""
        return self._prev_nodes

    @property
    def next_nodes(self):
        """Next nodes of the node."""
        return self._next_nodes

    def add_next_node(self, node: "Node"):
        """Add next node to the node.

        Args:
            node (Node): Next node to be added.
        """
        self._next_nodes.append(node)

    def add_prev_node(self, node: "Node"):
        """Add previous node to the node.

        Args:
            node (Node): Previous node to be added.
        """
        self._prev_nodes.append(node)

    def get_all(
        self,
        visited: Optional[Sequence["Node"]] = None,
    ) -> Sequence["Node"]:
        """Get all nodes connected to the node.

        Args:
            visited (Optional[Sequence["Node"]]): Visited nodes.

        Returns:
            Sequence["Node"]: All nodes connected to the node.
        """
        if visited is None:
            visited = set()
        if self._name in visited:
            return []
        visited.add(self._name)
        nodes = [self]
        for node in self._next_nodes:
            nodes += node.get_all(visited)
        return nodes

    def flatten(self) -> Sequence[Mapping[str, Any]]:
        """Flatten the node.

        Returns:
            Sequence[Mapping[str, Any]]: Flattened node.
        """
        return [
            {
                "name": node.name,
                "value_dict": node.value_dict,
                "prev_nodes": [n.name for n in node.prev_nodes],
                "next_nodes": [n.name for n in node.next_nodes],
                "is_end": node.is_end,
            }
            for node in self.get_all()
        ]
