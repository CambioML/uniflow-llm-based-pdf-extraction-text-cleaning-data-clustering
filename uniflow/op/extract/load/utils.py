"""Defines a function to read files from Amazon S3, URLs, or local paths in text or binary mode."""


def read_file(source: str, mode: str = "r"):
    """
    Reads a file from an S3 bucket, a URL, or the local filesystem based on the source path provided.

    Args:
        source (str): The path to the file. This can be an S3 bucket path (s3://bucket_name/file_path),
            a URL (http:// or https://), or a local file path.
        mode (str): The mode in which to open the file. Use 'r' for text mode and 'rb' for binary mode.
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

    # Read file from S3
    if source.startswith("s3://"):
        try:
            import boto3  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Please install boto3. You can use `pip install boto3` to install it."
            ) from exc

        bucket_name, file_path = source[5:].split("/", 1)
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket_name, Key=file_path)
        data = obj["Body"].read()
        return data.decode() if mode == "r" else data

    # Read file from URL
    if source.startswith("http://") or source.startswith("https://"):
        import requests  # pylint: disable=import-outside-toplevel

        # Create a session object
        session = requests.Session()

        # Optionally set headers to mimic a browser request
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }
        )

        # Use the session to make a request
        response = session.get(source, timeout=300)

        # Check if the request was successful
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("text/html"):
            raise ValueError(f"Expected content type text/html. Got {content_type}.")

        return response.text if mode == "r" else response.content

    # Read file from local filesystem
    with open(source, mode, encoding="utf-8") as file:
        return file.read()
