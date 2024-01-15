import psycopg2
import threading


db = psycopg2.connect(dbname="uniflow")
lock = threading.Lock()


def get_with_page(page: int) -> dict:
    """
    Synchronous get key and value in single page
    """
    lock.acquire()
    cur = db.cursor()
    # TODO: make offset smaller than rows in db
    offset = (page - 1) * 50
    query = f"""
    select key, value
    from value_dict
    limit 50
    offset {offset}
    ;"""
    cur.execute(query)
    cur.close()
    lock.release()

    return dict(cur.fetchall())
