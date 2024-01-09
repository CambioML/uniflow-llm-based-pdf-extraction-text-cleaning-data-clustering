"""Extract HTML op."""
import copy
from typing import Sequence

from uniflow.node import Node
from uniflow.op.op import Op

class ExtractHTMLOp(Op):
    """Extract HTML Op Class."""

    def __call__(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Op.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        output_nodes = []
        for node in nodes:
            value_dict = copy.deepcopy(node.value_dict)
            if "url" in value_dict:
                import requests
                resp = requests.get(
                    url=value_dict["url"]
                )
                text = resp.text
            else:
                with open(
                    value_dict["filename"],
                    "r",
                    encoding=value_dict.get("encoding", "utf-8"),
                ) as f:
                    text = f.read()
            text = self.parse_html(text)
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict={"text": text},
                    prev_nodes=[node],
                )
            )
        return output_nodes
    
    def parse_html(self, text):
        try:
            import bs4  # pylint: disable=import-outside-toplevel
            from bs4 import BeautifulSoup  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Please install bs4. You can use `pip install bs4` to install them."
            ) from exc
        
        soup = BeautifulSoup(text, 'html.parser')

        if soup.title:
            title = str(soup.title.string)
        else:
            title = ""

        ret = []
        for x in soup.body.descendants:
            node = x.name
            text = x.string
            if node != None and text != None and text != "":
                ret.append(f"{text.strip()}")
        return (title + "\n".join(ret))

class ProcessHTMLOp(Op):
    """Process HTML Op Class."""

    def __call__(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Op.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        output_nodes = []
        for node in nodes:
            value_dict = copy.deepcopy(node.value_dict)
            text = value_dict["text"]
            text = text.strip()
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict={"text": text},
                    prev_nodes=[node],
                )
            )
        return output_nodes