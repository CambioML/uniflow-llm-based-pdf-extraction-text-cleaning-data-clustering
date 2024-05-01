"""Recursive character split op."""

import copy
import re
from typing import Iterable, List, Sequence

import tiktoken

from uniflow.node import Node
from uniflow.op.op import Op


class RecursiveCharacterSplitter(Op):
    """Recursive character splitter class."""

    default_chunk_size = 1024
    default_chunk_overlap_size = 32
    default_separators = "\n\n|\n|. |.|, | |"
    default_splitting_mode = "char"

    def __init__(
        self,
        splitterConfig: dict,
        name: str = "recursive_character_splitter_op",
    ) -> None:
        """Recursive Splitter Op Constructor

        This has the effect of trying to keep all paragraphs (and then sentences, and then words) together
        as long as possible, as those would generically seem to be the strongest semantically related pieces of text.

        Args:
            name (str): Name of the op.
            chunk_size (int): Maximum size of chunks to return.
            chunk_overlap_size (int): Overlap in characters between chunks.
            separators (List[str]): Separators to use.
            splitting_mode (str): "char" for character count, "token" for token count. Defaults to "char".
            keep_separator (bool): Whether to keep the separator. Defaults to True.
            is_separator_regex (bool): Whether the separator is a regex. Defaults to False.
        """
        super().__init__(name)

        # Set up the splitter configuration
        self._chunk_size = splitterConfig["max_chunk_size"] or self.default_chunk_size
        self._separators = (
            ("separators" in splitterConfig and splitterConfig["separators"])
            or self.default_separators
        ).split("|")
        print(f"Separators: {self._separators}")

        # Set up the splitter configuration for recursive splitting
        self._chunk_overlap_size = (
            "chunk_overlap_size" in splitterConfig
            and splitterConfig["chunk_overlap_size"]
        ) or self.default_chunk_overlap_size
        self._splitting_mode = (
            "splitting_mode" in splitterConfig and splitterConfig["splitting_mode"]
        ) or self.default_splitting_mode
        self._keep_separator = (
            True
            and ("keep_separator" in splitterConfig)
            and splitterConfig["keep_separator"]
        )
        self._is_separator_regex = (
            ("is_separator_regex" in splitterConfig)
            and splitterConfig["is_separator_regex"]
            or False
        )

        self._encoder = tiktoken.encoding_for_model(
            "gpt-3.5"
        )  # Setup encoder for token-based splitting

    def _get_length(self, text: str) -> int:
        """Return the length of the text, either in characters or tokens."""
        if self._splitting_mode == "token":
            return len(self._encoder.encode(text))
        return len(text)

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
        cur_separator = (
            cur_separator if self._is_separator_regex else re.escape(cur_separator)
        )
        splits = [s for s in re.split(cur_separator, text) if s != ""]

        # Then go merging things, recursively splitting longer texts.
        _tmp_splits = []
        merge_separator = "" if self._keep_separator else _separator
        for s in splits:
            if self._get_length(s) <= self._chunk_size:
                _tmp_splits.append(s)
            else:
                # merge splitted texts into a chunk
                if _tmp_splits:
                    merged_text = self._merge_splits(_tmp_splits, merge_separator)
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
            merged_text = self._merge_splits(_tmp_splits, merge_separator)
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
        separator_len = self._get_length(separator)  # Use _get_length for separator

        docs, total = [], 0
        current_doc: List[str] = []
        for s in splits:
            _len = self._get_length(s)  # Use _get_length for split length
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
                        popped_length = self._get_length(
                            current_doc[0]
                        )  # Adjust pop length calculation
                        total -= popped_length + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]

            current_doc.append(s)
            total += _len + (separator_len if len(current_doc) > 1 else 0)

        doc = separator.join(current_doc).strip()

        if doc is not None:
            docs.append(doc)

        return docs
