"""Extract Gmail flow."""

from uniflow.constants import EXTRACT
from uniflow.flow.flow import Flow
from uniflow.op.extract.load.gcp.workspace.gmail_op import GmailOp


class ExtractGmailFlow(Flow):
    """Extract Gmail Flow Class."""

    TAG = EXTRACT

    def __init__(
        self,
        credentials_path: str = "",
        token_path: str = "",
    ):
        """Extract Gmail Flow Constructor."""
        super().__init__()
        self._gmail_op = GmailOp(
            name="gmail_op", credentials_path=credentials_path, token_path=token_path
        )

    def run(self, nodes):
        """Run Extract Gmail Flow.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        return self._gmail_op(nodes)
