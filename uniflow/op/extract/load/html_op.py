"""Extract HTML op."""

import copy
from typing import Sequence

from uniflow.node import Node
from uniflow.op.op import Op


class ExtractHTMLOp(Op):
    """Extract HTML Op Class."""

    def __init__(self, name: str) -> None:
        try:
            import requests  # pylint: disable=import-outside-toplevel
            from bs4 import BeautifulSoup  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Please install bs4. You can use `pip install bs4` to install them."
            ) from exc

        super().__init__(name)
        self._requests_client = requests
        self._beautiful_soup_parser = BeautifulSoup

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
                resp = self._requests_client.get(url=value_dict["url"], timeout=300)
                if not resp.ok:
                    raise ValueError(f"URL return an error: {resp.status_code}")

                content_type = resp.headers.get("Content-Type", "")
                if not content_type.startswith("text/html"):
                    raise ValueError(
                        f"Expected content type text/html. Got {content_type}."
                    )

                text = resp.text

            elif "filename" in value_dict:
                with open(
                    value_dict["filename"],
                    "r",
                    encoding=value_dict.get("encoding", "utf-8"),
                ) as f:
                    text = f.read()

            else:
                raise ValueError("Expected url or filename param.")

            text = self._parse_html(text)
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict={"text": text},
                    prev_nodes=[node],
                )
            )
        return output_nodes

    def _parse_html(self, text: str) -> str:
        """Function Parse Html.

        Args:
            text (str): Raw html text.

        Returns:
            str: Parsed html text.
        """
        soup = self._beautiful_soup_parser(text, "html.parser")

        if soup.title:
            title = str(soup.title.string) + "\n\n"
        else:
            title = ""

        return title + "\n\n".join(soup.body.stripped_strings)


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
