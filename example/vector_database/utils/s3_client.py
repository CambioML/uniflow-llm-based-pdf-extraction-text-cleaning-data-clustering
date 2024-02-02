import os
import warnings
from typing import Any, Dict, List, Optional


class S3Client:
    """
    Creates an S3 client using the specified session
    """

    def __init__(self, aws_session, loader_config: Dict[str, Any]) -> None:
        self._aws_region = (
            loader_config["aws_region"] if "aws_region" in loader_config else None
        )

        try:
            # import in class level to avoid installing boto3
            import boto3

            self._s3_client = aws_session.client("s3", region_name=self._aws_region)

        except ImportError as e:
            raise ModuleNotFoundError(
                "Failed to import the 'boto3' Python package. "
                "Please install it by running `pip install boto3`."
            ) from e
        except Exception as e:
            raise ValueError("Failed to create S3 client.") from e

    def download_file_from_s3(self, s3_bucket, s3_key, local_file_dirname="s3_files"):
        """
        Downloads a file from an S3 bucket.

        Args:
            s3_bucket (str): The name of the S3 bucket.
            s3_key (str): The key of the file in the S3 bucket.
            local_file_path (str): The local file path to download the file to.
        """
        file_basename = os.path.basename(s3_key)
        os.makedirs(local_file_dirname, exist_ok=True)
        local_file_path = os.path.join(local_file_dirname, file_basename)
        print(f"Downloading file from S3 to {local_file_path}")
        self._s3_client.download_file(s3_bucket, s3_key, local_file_path)

        return local_file_path

    def download_files_from_s3(
        self, s3_bucket, s3_prefix, local_file_dirname="s3_files"
    ):
        """
        Load the content of files in the specified S3 bucket and prefix.

        Args:
            s3_bucket (str): The name of the S3 bucket.
            s3_prefix (str): The prefix of the S3 objects.

        Yields:
            str: The content of each file in the S3 bucket and prefix.

        """
        paginator = self._s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)

        local_file_paths = []

        for page in page_iterator:
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                # Skip directories which end with "/"
                if s3_key.endswith("/"):
                    continue
                else:
                    local_file_path = self.download_file_from_s3(
                        s3_bucket, s3_key, local_file_dirname
                    )
                    local_file_paths.append(local_file_path)

        return local_file_paths

    def upload_file_to_s3(self, s3_bucket, s3_key, local_file_path):
        """
        Uploads a file to an S3 bucket.

        Args:
            s3_bucket (str): The name of the S3 bucket.
            s3_key (str): The key of the file in the S3 bucket.
            local_file_path (str): The local file path to upload the file from.
        """
        print(f"Uploading file to S3 from {local_file_path}")
        s3_path = f"s3://{s3_bucket}/{s3_key}"
        self._s3_client.upload_file(local_file_path, s3_bucket, s3_key)
        return s3_path

    def upload_files_to_s3(self, s3_bucket, s3_prefix, local_file_dirname):
        """
        Uploads files to an S3 bucket.

        Args:
            s3_bucket (str): The name of the S3 bucket.
            s3_prefix (str): The prefix of the S3 objects.
            local_file_path (str): The local file path to upload the file from.
        """
        s3_paths = []
        for file in os.listdir(local_file_dirname):
            local_file = os.path.join(local_file_dirname, file)
            s3_key = os.path.join(s3_prefix, file)
            self.upload_file_to_s3(s3_bucket, s3_key, local_file)
            s3_paths.append(f"s3://{s3_bucket}/{s3_key}")

        return s3_paths
