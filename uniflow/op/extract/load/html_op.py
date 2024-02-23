"""Extract HTML op."""

import copy
from typing import List, Sequence

from uniflow.node import Node
from uniflow.op.extract.load.utils import read_file
from uniflow.op.op import Op

TEXT_TAGS: List[str] = ["p", "a", "td", "span", "font"]
LIST_ITEM_TAGS: List[str] = ["li", "dd"]
LIST_TAGS: List[str] = ["ul", "ol", "dl"]
HEADING_TAGS: List[str] = ["h1", "h2", "h3", "h4", "h5", "h6"]
TABLE_TAGS: List[str] = ["table", "tbody", "td", "tr"]
TEXTBREAK_TAGS: List[str] = ["br"]
PAGEBREAK_TAGS: List[str] = ["hr"]
EMPTY_TAGS: List[str] = PAGEBREAK_TAGS + TEXTBREAK_TAGS
HEADER_OR_FOOTER_TAGS: List[str] = ["header", "footer"]
SECTION_TAGS: List[str] = ["div", "pre"]


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
                text = read_file(value_dict["url"])
            elif "filename" in value_dict:
                text = read_file(value_dict["filename"])
            else:
                raise ValueError("Expected url or filename param.")

            text = self._parse_html_from_element(text)
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict={"text": text},
                    prev_nodes=[node],
                )
            )
        return output_nodes

    def _is_container(self, tag_elem):
        """Checks if a tag is a container that also happens to contain text.

        Example
        -------
        <div>Hi, this is a container
            <span>This is a text span in container</span>
        </div>
        """
        if tag_elem.name not in (SECTION_TAGS + ["body"]) or len(tag_elem) == 0:
            return False

        return True

    def _parse_html_from_element(self, text: str) -> str:
        """Parse html from element by rules.

        Args:
            text (str): Raw html text.

        Returns:
            str: Parsed html text.
        """
        soup = self._beautiful_soup_parser(text, "html.parser")

        ret, descendanttag_elems = [], []
        for tag_elem in soup.body.descendants:
            tmp = ""

            # Prevent repeat tag
            if tag_elem in descendanttag_elems:
                continue

            # Text tag
            if tag_elem.name in (TEXT_TAGS + HEADING_TAGS + TEXTBREAK_TAGS):
                if len(list(tag_elem.children)) == 0:
                    continue

                descendanttag_elems = list(tag_elem.descendants)

                tmp = (" ").join(
                    [
                        p.replace("\n", " ")
                        for p in tag_elem.stripped_strings
                        if p.strip() != ""
                    ]
                )

            # Container
            elif self._is_container(tag_elem):
                # Container without text
                # E.g. <div><span>aaa</span<div>
                if (tag_elem.string is None or tag_elem.string.strip() == "") and len(
                    list(tag_elem.children)
                ) > 0:
                    # descendanttag_elems = list(tag_elem.children)
                    continue

                # Container with text
                # E.g. <div>aaa<span>bbb</div>
                else:
                    descendanttag_elems = list(tag_elem.descendants)

                    tmp = ("\n").join(
                        [p for p in tag_elem.stripped_strings if p.strip() != ""]
                    )

            # Merge table and list text
            elif tag_elem.name in (TABLE_TAGS + LIST_TAGS):
                tmp = ("\n").join(tag_elem.stripped_strings)
                descendanttag_elems = list(tag_elem.descendants)

            # Filter short content
            if tmp and tmp.strip() != "" and len(tmp.split(" ")) > 1:
                ret.append(tmp)

        return ("\n\n").join(ret)

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
