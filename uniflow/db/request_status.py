import psycopg2
import threading

from typing import Literal


db = psycopg2.connect(dbname="uniflow")
lock = threading.Lock()


def set_status(
    id: int,
    status: Literal['pending', 'running', 'complete', 'failed']
) -> None:
    """
    Asynchronous execution, add flow task status to db
    """
    # TODO: if 'complete' come faster than previous status
    # get_status will return the latest row which might not same as actual (latest status)
    # easy to fix, pass in 'time' instead generate it in DB, but dont have time
    cur = db.cursor()
    query = f"""
    insert into request_status
    values('{id}', '{status}')
    ;"""
    cur.execute(query)
    db.commit()
    cur.close()


def get_status(id: int) -> str:
    """
    Synchronous get status of flow run.
    """
    lock.acquire()
    cur = db.cursor()
    query = f"""
    select *
    from request_status
    where id = '{id}'
    order by _time desc
    limit 1
    ;"""
    cur.execute(query)
    result = cur.fetchone()
    # TODO
    # if (result is None):
    #     raise KeyError
    db.commit()
    cur.close()
    lock.release()

    return result[1]
