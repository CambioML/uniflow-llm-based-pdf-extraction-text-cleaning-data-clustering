from typing import Any, Mapping, Sequence
from flask import Flask, request, jsonify
import uuid
import sqlite3
import json
import asyncio
from math import ceil

import sys

sys.path.append("..")

from uniflow.flow.client import TransformClient
from uniflow.flow.config import ExpendReduceConfig
from uniflow.flow.database import Database

app = Flask(__name__)


async def async_start_flow(input: Sequence[Mapping[str, Any]], job_id: str) -> None:
    """Start the flow asynchronously.

    Args:
        input (Sequence[Mapping[str, Any]]): the input data
        job_id (str): the job id
    """
    try:
        client = TransformClient(ExpendReduceConfig())
        client.run(input)
        Database().update_job_status(job_id, "completed")
    except ValueError as e:
        Database().update_job_status(job_id, "failed")


@app.route("/flows/expand_reduce", methods=["POST"])
def start_flow():
    """Start the flow asynchronously.

    Returns:
        the job id
    """
    try:
        input_data = request.get_json()
        if not isinstance(input_data, list) or not all(
            isinstance(item, dict) for item in input_data
        ):
            return (
                jsonify(
                    {"error": "Invalid input format. Expected a list of dictionaries."}
                ),
                400,
            )

        job_id = str(uuid.uuid4())
        Database().update_job_status(job_id, "pending")
        asyncio.run(async_start_flow(input_data, job_id))

        return jsonify({"job_id": job_id}), 202
    except json.JSONDecodeError:
        return jsonify({"error": "Bad request"}), 400


@app.route("/flows/status/<job_id>")
def get_job_status(job_id):
    """Get the status of the job.

    Args:
        job_id (str): the job id

    Returns:
        the status of the job
    """
    status = Database().get_job_status(job_id)

    if status:
        return jsonify({"job_id": job_id, "status": status})
    else:
        return jsonify({"error": "Job not found"}), 404


@app.route("/flows/results")
def get_all_results():
    """Get all results in the database.

    Returns:
        all key value pairs in the database
    """
    try:
        page = request.args.get("page", 1, type=int)
        limit = request.args.get("limit", 10, type=int)
        offset = (page - 1) * limit
        results = Database().select_all(limit=limit, offset=offset)
        result_dicts = {}
        for result in results:
            result_dicts[result[0]] = result[1]
        total = Database().count_all()
        return jsonify(
            {"results": result_dicts, "total_pages": ceil(1.0 * total / limit)}
        )
    except ValueError:
        return jsonify({"error": "Bad request"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
