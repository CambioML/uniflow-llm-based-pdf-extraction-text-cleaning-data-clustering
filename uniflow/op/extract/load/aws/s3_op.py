"""S3 Op Class."""

import copy
import logging
import os
from typing import Sequence

from uniflow.node import Node
from uniflow.op.op import Op

logger = logging.getLogger(__name__)


class ExtractS3Op(Op):
    """Op to download a file from s3."""

    LOCAL_FILE_PATH = "/tmp/aws/s3"

    def __init__(self, name: str = "extract_s3_op") -> None:
        try:
            import boto3  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError("Please install boto3 to use S3Op.") from e

        self._s3_client = boto3.client("s3")
        super().__init__(name=name)

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
            # create local file path if not exists
            if os.path.exists(self.LOCAL_FILE_PATH) is False:
                os.makedirs(self.LOCAL_FILE_PATH)
            filename = os.path.join(self.LOCAL_FILE_PATH, value_dict["key"])
            logger.info("Downloading %s to %s", value_dict["key"], filename)
            self._s3_client.download_file(
                Bucket=value_dict["bucket"],
                Key=value_dict["key"],
                Filename=filename,
            )
            with open(
                filename,
                "r",
                encoding=value_dict.get("encoding", "utf-8"),
            ) as f:
                text = f.read()
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict={"text": text},
                    prev_nodes=[node],
                )
            )
        return output_nodes
