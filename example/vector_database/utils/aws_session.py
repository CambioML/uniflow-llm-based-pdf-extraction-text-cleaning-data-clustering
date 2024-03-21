import warnings
from typing import Any, Dict


class AWSSession:
    """Creates a boto3 session using the specified profile name or AK/SK."""

    def __init__(self, loader_config: Dict[str, Any]) -> None:
        try:
            # import in class level to avoid installing boto3
            import boto3

            # If user specifies profile in model config, use that profile
            if "aws_profile" in loader_config:
                aws_profile = loader_config["aws_profile"]
                self.session = boto3.Session(profile_name=aws_profile)
            # Otherwise if the user specifies credentials directly in the model config, use those credentials
            elif loader_config.get("aws_access_key_id") and loader_config.get(
                "aws_secret_access_key"
            ):
                self.session = boto3.Session(
                    aws_access_key_id=loader_config["aws_access_key_id"],
                    aws_secret_access_key=loader_config["aws_secret_access_key"],
                    aws_session_token=loader_config.get("aws_session_token"),
                )
                warnings.warn(
                    "Using AWS credentials directly in the model config is not recommended. "
                    "Please use a profile instead."
                )
            else:
                self.session = boto3.Session(profile_name="default")
                warnings.warn(
                    "Using default profile to create the session. "
                    "Please pass the profile name in the model config."
                )

        except ImportError as e:
            raise ModuleNotFoundError(
                "Failed to import the 'boto3' Python package. "
                "Please install it by running `pip install boto3`."
            ) from e
        except Exception as e:
            raise ValueError(
                "Failed to load credentials for authenticating with the AWS client. "
                "Please ensure that the specified profile name contains valid credentials."
            ) from e
