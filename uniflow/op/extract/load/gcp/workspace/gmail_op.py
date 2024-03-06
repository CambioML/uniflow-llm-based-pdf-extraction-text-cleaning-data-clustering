"""Gmail Op Module."""

import base64
import re
from typing import Sequence

from uniflow.node import Node
from uniflow.op.op import Op

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
SPAM_LABEL = "Spam Email (AI Email Filter)"
NON_SPAM_LABEL = "Email (AI Email Filter)"


class GmailOp(Op):
    """Gmail Op Class."""

    def __init__(
        self,
        name: str,
        credentials_path: str = "",
        token_path: str = "",
    ) -> None:
        """Gmail Filter Flow Constructor.

        Follow https://developers.google.com/gmail/api/quickstart/python to get credentials.json and token.json.

        Args:
            name (str): Op name.
            credentials_path (str): Path to google credentials.json.
            token_path (str): Path to google token.json.

        """
        try:
            from googleapiclient.discovery import (  # pylint: disable=import-outside-toplevel
                build,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Please install below packages. You can use "
                "`pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib` "
                "to install it.",
            ) from exc

        # create google workspace token.
        credentials = self._create_token(credentials_path, token_path)
        # create google workspace service client.
        self._service = build("gmail", "v1", credentials=credentials)
        # create label.
        self._create_label()

        super().__init__(name)

    def _create_token(self, credentials_path: str, token_path: str):
        """Create token."""
        try:
            import os.path

            from google.auth.transport.requests import (  # pylint: disable=import-outside-toplevel
                Request,
            )
            from google.oauth2.credentials import (  # pylint: disable=import-outside-toplevel
                Credentials,
            )
            from google_auth_oauthlib.flow import (  # pylint: disable=import-outside-toplevel
                InstalledAppFlow,
            )

            # from googleapiclient.errors import (  # pylint: disable=import-outside-toplevel
            #     HttpError,
            # )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Please install below packages. You can use "
                "`pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib` "
                "to install it.",
            ) from exc

        if not os.path.exists(credentials_path) and not os.path.exists(token_path):
            raise FileNotFoundError(
                "Please download credentials.json from https://developers.google.com/gmail/api/quickstart/python "
                "and save it in the current directory.",
            )

        credentials = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists(token_path):
            credentials = Credentials.from_authorized_user_file(token_path, SCOPES)

        # If there are no (valid) credentials available, let the user log in.
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                app_flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, SCOPES
                )
                credentials = app_flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(token_path, "w") as token:
                token.write(credentials.to_json())

        return credentials

    def _create_label(self):
        """Create label."""
        spam_label = {
            "name": SPAM_LABEL,
            "messageListVisibility": "show",
            "labelListVisibility": "labelShow",
        }
        non_spam_label = {
            "name": NON_SPAM_LABEL,
            "messageListVisibility": "show",
            "labelListVisibility": "labelShow",
        }
        results = self._service.users().labels().list(userId="me").execute()
        labels = [label["name"] for label in results.get("labels", [])]

        if SPAM_LABEL not in labels:
            self._service.users().labels().create(
                userId="me", body=spam_label
            ).execute()
        if NON_SPAM_LABEL not in labels:
            self._service.users().labels().create(
                userId="me", body=non_spam_label
            ).execute()

    def _get_email(self):
        output = []
        # Get the 10 latest unread Gmail threads
        threads = (
            self._service.users()
            .threads()
            .list(userId="me", q="is:unread", maxResults=10)
            .execute()
            .get("threads", [])
        )

        # spam_email_id = self._get_label_id(SPAM_LABEL)
        # non_spam_email_id = self._get_label_id(NON_SPAM_LABEL)

        for thread in threads:
            t = (
                self._service.users()
                .threads()
                .get(userId="me", id=thread["id"])
                .execute()
            )

            # # Automatically archive if it already filtered
            # if (
            #     spam_email_id in t["messages"][0]["labelIds"]
            #     or non_spam_email_id in t["messages"][0]["labelIds"]
            # ):
            #     continue

            # Get the last message from the thread
            last_message = t["messages"][-1]

            from_address = ""
            for header in last_message["payload"]["headers"]:
                if header["name"] == "From":
                    from_address = header["value"]
                    break

            if not self._is_address_emailed(from_address):
                email_body = self._get_plain_body(last_message)
                body = base64.urlsafe_b64decode(email_body.encode("ASCII"))

                output.append(
                    {
                        "from": from_address,
                        "body": body,
                        "email_id": thread["id"],
                        "snippet": last_message["snippet"],
                    }
                )
        return output

    def _get_plain_body(self, message):
        """Get plain body."""
        parts = message["payload"].get("parts", [])
        for part in parts:
            if part["mimeType"] == "text/plain":
                return part["body"]["data"]
        return ""

    def _get_label_id(self, label_name: str):
        """Get email label."""
        labels = (
            self._service.users().labels().list(userId="me").execute().get("labels", [])
        )
        for label in labels:
            if label["name"] == label_name:
                return label["id"]
        return None

    def _is_address_emailed(self, email_address: str):
        """Check if email address is emailed."""

        def extract_content_between_angle_brackets(s):
            match = re.search(r"<([^>]*)>", s)
            return match.group(1) if match else s

        email_address = extract_content_between_angle_brackets(email_address)
        query = f"to:{email_address} in:sent"
        threads = (
            self._service.users()
            .threads()
            .list(userId="me", q=query)
            .execute()
            .get("threads", [])
        )
        return len(threads) > 0

    def __call__(self, nodes: Sequence[Node]) -> Sequence[Node]:
        """Run Model Op.

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running.
        """
        output_nodes = []
        for node in nodes:
            output = self._get_email()
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict=output,
                    prev_nodes=[node],
                )
            )
        return output_nodes
