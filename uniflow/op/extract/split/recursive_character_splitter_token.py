"""Recursive character split op."""

import copy
import re
from typing import Iterable, List, Optional, Sequence

import tiktoken
from uniflow.node import Node
from uniflow.op.op import Op


class RecursiveCharacterSplitter_Token(Op):
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
        self._encoder = tiktoken.encoding_for_model("gpt-3.5")

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
        """Recursively split the text into chunks based on token length.

        Args:
            text (str): Text to split.
            separators (List[str]): Separators for splitting.

        Returns:
            List[str]: Chunks after splitting.
        """
        final_chunks, next_separators = [], []

        if len(separators) == 0:
            return final_chunks

        # Find the current separator and the next separators to use for splitting
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

        # Split the text using the current separator
        cur_separator = re.escape(cur_separator)
        splits = [s for s in re.split(cur_separator, text) if s != ""]

        # Process each split
        _tmp_splits, _separator = [], ""
        for s in splits:
            # Calculate the token length of the current split
            token_len = len(self._encoder.encode(s))
            # Check if the token length is within the chunk size
            if token_len < self._chunk_size:
                _tmp_splits.append(s)
            else:
                # Merge and add the temporary splits to the final chunks
                if _tmp_splits:
                    merged_text = self._merge_splits(_tmp_splits, _separator)
                    final_chunks.extend(merged_text)
                    _tmp_splits = []

                # Recursively split the current split if it's too long and there are more separators to use
                if not next_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._recursive_splitter(s, next_separators)
                    final_chunks.extend(other_info)

        # Merge and add any remaining temporary splits to the final chunks
        if _tmp_splits:
            merged_text = self._merge_splits(_tmp_splits, _separator)
            final_chunks.extend(merged_text)

        return final_chunks

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        """Combine smaller pieces into medium-sized chunks based on token length.

        Args:
            splits (Iterable[str]): Smaller pieces before merge.
            separator (str): Separator for merge.

        Returns:
            List[str]: Merged medium-sized chunks.
        """
        docs, total = [], 0
        current_doc: List[str] = []
        for s in splits:
            # Calculate the token length of the current split
            token_len = len(self._encoder.encode(s))
            # Calculate the total token length including the separator
            current_length = (
                total + token_len + (len(self._encoder.encode(separator)) if len(current_doc) > 0 else 0)
            )

            # Check if the current length exceeds the chunk size
            if current_length > self._chunk_size:
                # Warn if the total token length of the current document exceeds the chunk size
                if total > self._chunk_size:
                    print(
                        f"Created a chunk of size {total} tokens, "
                        f"which is longer than the specified {self._chunk_size} tokens"
                    )
                # Add the current document to the list of documents if it's not empty
                if len(current_doc) > 0:
                    doc = separator.join(current_doc).strip()
                    if doc is not None:
                        docs.append(doc)
                    # Remove splits from the beginning of the current document until it meets the overlap size or length criteria
                    while total > self._chunk_overlap_size or (
                        current_length > self._chunk_size and total > 0
                    ):
                        total -= len(self._encoder.encode(current_doc[0])) + (
                            len(self._encoder.encode(separator)) if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]

            # Add the current split to the current document and update the total token length
            current_doc.append(s)
            total += token_len + (len(self._encoder.encode(separator)) if len(current_doc) > 1 else 0)

        # Add the remaining document to the list of documents
        doc = separator.join(current_doc).strip()
        if doc is not None:
            docs.append(doc)

        return docs

