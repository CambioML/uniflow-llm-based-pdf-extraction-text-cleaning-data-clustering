import sys
import unittest
from io import BytesIO
from unittest.mock import mock_open, patch

from uniflow.op.extract.load.utils import read_file


class TestReadFile(unittest.TestCase):
    def test_read_file_local(self):
        with patch(
            "builtins.open", mock_open(read_data="test_local_data")
        ) as mock_file:
            content = read_file("test_local.txt", mode="r")
            mock_file.assert_called_with("test_local.txt", "r", encoding="utf-8")
            self.assertEqual(content, "test_local_data")

    def test_read_file_s3(self):
        mock_body = BytesIO(b"test_s3_data")

        with patch("boto3.client") as mock_boto3:
            mock_boto3.return_value.get_object.return_value = {"Body": mock_body}
            content = read_file("s3://bucket/test_s3.txt", mode="r")
            mock_boto3.return_value.get_object.assert_called_with(
                Bucket="bucket", Key="test_s3.txt"
            )
            self.assertEqual(content, "test_s3_data")

    def test_read_file_url(self):
        mock_get_response = unittest.mock.Mock()
        mock_get_response.raise_for_status = unittest.mock.Mock()
        mock_get_response.text = "test_url_data"
        mock_get_response.content = b"test_url_data"
        mock_get_response.headers = {"Content-Type": "text/html"}

        with patch("requests.Session") as mock_session:
            mock_session.return_value.get.return_value = mock_get_response
            content = read_file("http://example.com/test_url.txt", mode="r")
            self.assertEqual(content, "test_url_data")

    def test_read_file_binary(self):
        mock_binary_data = b"binary_data"

        with patch(
            "builtins.open", mock_open(read_data=mock_binary_data)
        ) as mock_file, patch("io.BytesIO") as mock_bytes_io:
            mock_bytes_io.return_value = BytesIO(mock_binary_data)
            content = read_file("test_binary.txt", mode="rb")
            mock_file.assert_called_with("test_binary.txt", "rb", encoding="utf-8")
            self.assertEqual(content, mock_binary_data)

    def test_read_file_unsupported_mode(self):
        with self.assertRaises(ValueError):
            read_file("test.txt", mode="invalid_mode")

    def test_read_file_invalid_url(self):
        with self.assertRaises(ValueError):
            read_file("http://invalid_url.com/test.txt", mode="r")

    def test_boto3_not_installed(self):
        with patch.dict(sys.modules, {"boto3": None}):
            with self.assertRaises(ModuleNotFoundError):
                read_file("s3://bucket/test.txt", mode="r")


if __name__ == "__main__":
    unittest.main()
