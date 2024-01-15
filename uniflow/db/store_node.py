import psycopg2
import threading
from typing import Any

from uniflow.node import Node


db = psycopg2.connect(dbname="uniflow")
node_lock = threading.Lock()
edge_lock = threading.Lock()


# add more if needed, consistent with db
value_type_dict = {
    type(int(1)): "int",
    type(str("str")): "str",
    type(float(0.1)): "float",
}


def store_node(node: Node) -> None:
    node_lock.acquire()
    cur = db.cursor()
    nname = node.name
    query = f"""
    select * from nodes
    where nname = '{nname}'
    ;"""
    cur.execute(query)
    if len(cur.fetchall()) > 0:
        # node already exists, all work below done
        cur.close()
        node_lock.release()
        return

    query = f"""
    insert into nodes
    values ('{nname}', {node.is_end})
    ;"""
    cur.execute(query)

    # insert node values
    for key, value in node.value_dict.items():
        query = f"""
        insert into value_dict
        values ('{nname}', '{key}', '{str(value)}', '{value_type_dict[type(value)]}')
        ;"""
        cur.execute(query)
    db.commit()
    cur.close()
    node_lock.release()

def store_edge(node: Node, next: Node) -> None:
    edge_lock.acquire()
    cur = db.cursor()
    query = f"""
    select * from node_edges
    where node = '{node.name}' and next = '{next.name}'
    ;"""
    cur.execute(query)
    if len(cur.fetchall()) > 0:
        # edge already exists
        cur.close()
        node_lock.release()
        return

    query = f"""
    insert into node_edges
    values ('{node.name}', '{next.name}')
    ;"""
    cur.execute(query)
    cur.close()
    db.commit()
    edge_lock.release()
