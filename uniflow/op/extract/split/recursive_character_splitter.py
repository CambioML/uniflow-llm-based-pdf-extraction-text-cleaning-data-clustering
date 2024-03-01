"""Recursive character split op."""

import copy
import re
from typing import Iterable, List, Optional, Sequence

from uniflow.node import Node
from uniflow.op.op import Op


class RecursiveCharacterSplitter(Op):
    """Recursive character splitter class."""

    default_separators = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        name: str,
        chunk_size: int = 1024,
        chunk_overlap_size: int = 0,
        separators: Optional[List[str]] = None,
    ) -> None:
        """Recursive Splitter Op Constructor

        This has the effect of trying to keep all paragraphs (and then sentences, and then words) together
        as long as possible, as those would generically seem to be the strongest semantically related pieces of text.

        Args:
            name (str): Name of the op.
            chunk_size (int): Maximum size of chunks to return.
            chunk_overlap_size (int): Overlap in characters between chunks.
            separators (List[str]): Separators to use.
            keep_separator: Whether to keep the separator in the chunks.
        """
        super().__init__(name)
        self._chunk_size = chunk_size
        self._chunk_overlap_size = chunk_overlap_size
        self._separators = separators or self.default_separators

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
            text = self._recursive_splitter(text.strip(), self._separators)
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict={"text": text},
                    prev_nodes=[node],
                )
            )
        return output_nodes

    def _recursive_splitter(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks.

        It takes in the large text then tries to split it by the first character \n\n. If the first split by \n\n is
        still large then it moves to the next character which is \n and tries to split by it. If it is still larger
        than our specified chunk size it moves to the next character in the set until we get a split that is less than
        our specified chunk size. The default separators list is ["\n\n", "\n", " ", ""])

        Args:
            text (str): Text to split.
            separators(List[str]): separators for split.

        Returns:
            List[str]: Chunks after split.
        """
        final_chunks, next_separators = [], []

        if len(separators) == 0:
            return final_chunks

        # Get current and next separators
        cur_separator = separators[-1]
        for i, _s in enumerate(separators):
            _separator = re.escape(_s)
            if _s == "":
                cur_separator = _s
                break
            if re.search(_separator, text):
                cur_separator = _s
                next_separators = separators[(i + 1) :]
                break

        # Splited by current separator firstly
        cur_separator = re.escape(cur_separator)
        splits = [s for s in re.split(cur_separator, text) if s != ""]

        # Then go merging things, recursively splitting longer texts.
        _tmp_splits, _separator = [], ""
        for s in splits:
            if len(s) < self._chunk_size:
                _tmp_splits.append(s)
            else:
                # merge splitted texts into a chunk
                if _tmp_splits:
                    merged_text = self._merge_splits(_tmp_splits, _separator)
                    final_chunks.extend(merged_text)
                    # reset tmp_splits
                    _tmp_splits = []

                # recursively split using next separators
                if not next_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._recursive_splitter(s, next_separators)
                    final_chunks.extend(other_info)

        if _tmp_splits:
            merged_text = self._merge_splits(_tmp_splits, _separator)
            final_chunks.extend(merged_text)

        return final_chunks

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        """Combine these smaller pieces into medium size chunks.

        Args:
            splits (Iterable[str]): Smaller pieces before merge.
            separator (str): Separator for merge.

        Returns:
            List[str]: Merged medium size chunks.
        """
        separator_len = len(separator)

        docs, total = [], 0
        current_doc: List[str] = []
        for s in splits:
            _len = len(s)
            current_length = (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
            )

            if current_length > self._chunk_size:
                if total > self._chunk_size:
                    print(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = separator.join(current_doc).strip()
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap_size or (
                        current_length > self._chunk_size and total > 0
                    ):
                        total -= len(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]

            current_doc.append(s)
            total += _len + (separator_len if len(current_doc) > 1 else 0)

        doc = separator.join(current_doc).strip()

        if doc is not None:
            docs.append(doc)

        return docs
