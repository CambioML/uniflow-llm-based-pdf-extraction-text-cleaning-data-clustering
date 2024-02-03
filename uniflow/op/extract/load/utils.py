"""Defines a function to read files from Amazon S3, URLs, or local paths in text or binary mode."""

import requests


def read_file(source: str, mode: str = "r"):
    """
    Reads a file from an S3 bucket, a URL, or the local filesystem based on the source path provided.

    Parameters:
    - source (str): The path to the file. This can be an S3 bucket path (s3://bucket_name/file_path),
      a URL (http:// or https://), or a local file path.
    - mode (str): The mode in which to open the file. Use 'r' for text mode and 'rb' for binary mode.
      Default is 'r'.

    Returns:
    The content of the file as a string if mode is 'r', or as bytes if mode is 'rb'.

    Example usage:
    ```
    content = read_file("s3://my_bucket/my_file.txt")
    content = read_file("https://my_bucket.abc.com/my_file.txt")
    content = read_file("./my_file.txt")
    ```
    Note:
    - When reading from an S3 bucket, make sure you have the necessary credentials and permissions.
    - When reading from a URL, ensure that the URL is accessible and the file exists.
    - When reading from the local filesystem, provide the correct file path.

    """
    if mode not in ["r", "rb"]:
        raise ValueError(
            "Unsupported mode. Use 'r' for text mode or 'rb' for binary mode."
        )

    try:
        if source.startswith("s3://"):
            try:
                import boto3  # pylint: disable=import-outside-toplevel
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Please install boto3. You can use `pip install boto3` to install it."
                ) from exc
            # Read file from S3
            bucket_name, file_path = source[5:].split("/", 1)
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket_name, Key=file_path)
            data = obj["Body"].read()
            return data.decode() if mode == "r" else data
        if source.startswith("http://") or source.startswith("https://"):
            # Read file from URL
            response = requests.get(source)
            response.raise_for_status()
            return response.text if mode == "r" else response.content
        # Read file from local filesystem
        with open(source, mode) as file:
            return file.read()
    except (requests.RequestException, IOError) as e:
        # Handle network request errors or file I/O errors
        print(f"Error accessing {source}: {e}")
    except Exception as e:
        # Handle any other unexpected errors
        print(f"An unexpected error occurred: {e}")
