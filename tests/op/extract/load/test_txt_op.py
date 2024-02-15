import unittest

from moto import mock_s3

from uniflow.node import Node
from uniflow.op.extract.load.txt_op import ExtractTxtOp, ProcessTxtOp


class TestExtractTxtOp(unittest.TestCase):
    def setUp(self):
        self.extract_txt_op = ExtractTxtOp(name="extract_txt_op")

    def test_empty_input(self):
        # arrange
        node = Node(
            name="node1",
            value_dict={"filename": "tests/op/extract/data/empty.txt"},
        )

        # act
        output_nodes = self.extract_txt_op([node])

        # assert
        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict["text"], "")

    def test_load_from_local(self):
        node = Node(
            name="node1",
            value_dict={"filename": "tests/op/extract/data/test.txt"},
        )

        output_nodes = self.extract_txt_op([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict["text"], "This is a test file.")

    @mock_s3
    def test_load_from_s3(self):
        import boto3  # pylint: disable=import-outside-toplevel

        conn = boto3.resource("s3", region_name="us-west-2")
        conn.create_bucket(Bucket="uniflow-test")
        s3 = boto3.client("s3", region_name="us-west-2")
        s3.put_object(Bucket="uniflow-test", Key="test.txt", Body="mycontent")

        node = Node(
            name="node1",
            value_dict={"filename": "s3://uniflow-test/test.txt"},
        )

        output_nodes = self.extract_txt_op([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict["content"], "mycontent")


class TestProcessTxtOp(unittest.TestCase):
    def setUp(self):
        self.process_txt_op = ProcessTxtOp(name="process_txt_op")

    def test_empty_input(self):
        node = Node(name="node1", value_dict={"text": ""})

        output_nodes = self.process_txt_op([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict["text"], [])

    def test_whitespace_input(self):
        node = Node(name="node1", value_dict={"text": " \n \n "})

        output_nodes = self.process_txt_op([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(output_nodes[0].value_dict["text"], [])

    def test_call(self):
        node = Node(
            name="node1", value_dict={"text": "This is a test\nThis is another test"}
        )

        output_nodes = self.process_txt_op([node])

        self.assertEqual(len(output_nodes), 1)
        self.assertEqual(
            output_nodes[0].value_dict["text"],
            ["This is a test", "This is another test"],
        )
