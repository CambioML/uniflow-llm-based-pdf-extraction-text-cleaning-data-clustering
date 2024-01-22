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
app.config['DATABASE'] = 'uniflow_data.db'

async def async_start_flow(input: str, job_id: str) -> None:
    client = TransformClient(ExpendReduceConfig())
    client.run(input)
    db = Database()
    db.update_job(job_id, 'completed')
    return
    
@app.route('/flows/expand_reduce', methods=['POST'])
def start_flow():
    job_id = str(uuid.uuid4())
    input_data = request.get_json()
    db = Database()
    db.insert_job(job_id, 'pending')
    asyncio.run(async_start_flow([input_data], job_id))

    return jsonify({'job_id': job_id}), 202

@app.route('/flows/status/<job_id>')
def get_job_status(job_id):
    db = Database()
    status = db.get_job_status(job_id)

    if status:
        return jsonify({'job_id': job_id, 'status': status})
    else:
        return jsonify({'error': 'Job not found'}), 404

@app.route('/flows/results')
def get_job_results():
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 10, type=int)
    offset = (page - 1) * limit
    db = Database()
    results = db.select_all(limit=limit, offset=offset)
    total = db.count_all()
    return jsonify({'results': results, 'total': total, 'total_pages': ceil(1.0 * total / limit)})

if __name__ == '__main__':
    app.run()