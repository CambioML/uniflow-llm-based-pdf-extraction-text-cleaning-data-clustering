"""Module for visualizing the results of the computational graph."""

from graphviz import Digraph


class Viz:
    """Visualize the results of the computational graph."""

    @staticmethod
    def to_digraph(self) -> Digraph:
        """Convert the node to graphviz Digraph.

        Returns:
            Digraph: Graphviz Digraph.
        """
        graph = Digraph()
        for node in self.get_all():
            graph.node(node.name)
            for next_node in node.next_nodes:
                graph.edge(node.name, next_node.name)
        return graph
