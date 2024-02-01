import warnings
from typing import Any, Dict, List, Optional

from requests_aws4auth import AWS4Auth


class ElasticSearchClient:
    """
    Creates an OpenSearch client using the specified opensearch_url.
    """

    def __init__(self, aws_session, loader_config: Dict[str, Any]) -> None:
        self._opensearch_url = loader_config["opensearch_url"]
        self._aws_region = (
            loader_config["aws_region"] if "aws_region" in loader_config else None
        )

        try:
            from opensearchpy import OpenSearch, RequestsHttpConnection

            if "es_username" in loader_config and "es_password" in loader_config:
                awsauth = (loader_config["es_username"], loader_config["es_password"])
            else:
                credentials = aws_session.get_credentials()

                awsauth = AWS4Auth(
                    credentials.access_key,
                    credentials.secret_key,
                    self._aws_region,
                    "es",
                    session_token=credentials.token,
                )

            self._elasticsearch_client = OpenSearch(
                hosts=[{"host": self._opensearch_url, "port": 443}],
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
            )
        except ImportError as e:
            raise ModuleNotFoundError(
                "Failed to import the 'opensearchpy' Python package. "
                "Please install it by running `pip install opensearch-py`."
            ) from e
        except Exception as e:
            raise ValueError(
                "Failed to create OpenSearch client."
                "Please ensure that the specified opensearch_url is valid."
            ) from e

    def check_index_exists(self, index_name: str) -> bool:
        """Checks if the index exists."""
        return self._elasticsearch_client.indices.exists(index_name)

    def create_index(self, index_name: str) -> None:
        """Creates an index."""

        default_index_template = {
            "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 32}},
            "mappings": {
                "properties": {
                    "vector_field": {
                        "type": "knn_vector",
                        "dimension": 1536,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {"ef_construction": 512, "m": 32},
                        },
                    }
                }
            },
        }
        self._elasticsearch_client.indices.create(
            index_name, body=default_index_template
        )

    def delete_index(self, index_name: str) -> None:
        """Deletes an index."""
        self._elasticsearch_client.indices.delete(index_name)

    def bulk_ingest_elasticsearch(
        self, index_name: str, ingest_data: List[Dict[str, Any]], is_aoss: bool = True
    ) -> None:
        """Bulk ingest data into OpenSearch."""

        from opensearchpy.exceptions import OpenSearchException
        from opensearchpy.helpers import bulk

        requests = []
        return_ids = []

        try:
            self._elasticsearch_client.indices.get(index=index_name)
        except OpenSearchException:
            self.create_index(index_name)

        for _, data in enumerate(ingest_data):
            metadata = data["metadata"]
            text = data["text"]
            _id = data["id"]
            embedding = data["vector_field"]

            request = {
                "_op_type": "index",
                "_index": index_name,
                "vector_field": embedding,
                "text": text,
                "metadata": metadata,
            }
            if is_aoss:
                request["id"] = _id
            else:
                request["_id"] = _id
            requests.append(request)
            return_ids.append(_id)

        bulk(self._elasticsearch_client, requests)

    def knn_search(
        self, index_name: str, embedding: List[float], k: int = 2
    ) -> List[int]:
        """Performs a KNN search on the given index and embedding."""
        from opensearchpy.exceptions import OpenSearchException

        try:
            search_query = {
                "size": k,
                "query": {"knn": {"vector_field": {"vector": embedding, "k": k}}},
            }
            response = self._elasticsearch_client.search(
                index=index_name, body=search_query
            )
        except OpenSearchException as e:
            raise ValueError(
                "Failed to perform KNN search on the given index and embedding."
            ) from e

        return [hit for hit in response["hits"]["hits"]]
