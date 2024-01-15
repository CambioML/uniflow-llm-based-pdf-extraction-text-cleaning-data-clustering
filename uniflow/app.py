"""
not fully tested
"""
import logging
import random
import threading
from json import dumps
from flask import Flask, request

from uniflow.node import Node
from uniflow.flow.expand_reduce_flow import ExpandReduceFlow
from uniflow.db.request_status import set_status, get_status
from uniflow.db.value_dict import get_with_page


logger = logging.getLogger(__name__)
APP = Flask(__name__)


def run_flow(id: int, data: dict):
    """thread run flow

    Args:
        id (int): task id.
        data (dict): input to puild a node.
    """
    root = Node(name="root", value_dict=data)
    flow = ExpandReduceFlow()
    set_status(id=id, status='running')
    flow.run(root)
    set_status(id=id, status='complete')


@APP.route("/api/v1/flow/expandReduce", methods=['Post'])
def expandReduce():
    task_id = random.randint(10000, 99999) # dummy id
    # data format: e.g. {"How are you?": "Fine."}
    data = request.get_json()
    x = threading.Thread(target=run_flow, args=(task_id, dict(data)))
    x.start()

    return dumps({
        'id': task_id,
    })


@APP.route("/api/v1/flow/expandReduce", methods=['Get'])
def expandReduceStatus():
    task_id = request.get_json()["id"]
    status = get_status(task_id)

    return dumps({
        'status': status,
    })


@APP.route("/api/v1/node/value", methods=['Get'])
def getAllNodeValues():
    page = request.get_json()["page"]

    return dumps(get_with_page(page=page))


if __name__ == '__main__':
    APP.run(debug=True)
