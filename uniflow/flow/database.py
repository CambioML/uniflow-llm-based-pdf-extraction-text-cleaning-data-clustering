"""Database class."""
import logging
from typing import Optional, Sequence, Mapping, Any
import sqlite3
from threading import Lock

logger = logging.getLogger(__name__)


class Database:
    """Database class."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Singleton: only create the Database instance once

        Args:
            cls (Database): the Database class"""
        if cls._instance is None:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Create and connect to database."""
        logging.basicConfig(format="%(levelname)s [%(module)s]: %(message)s")
        try:
            self._connection = sqlite3.connect("uniflow_data.db")
            self._cursor = self._connection.cursor()
            self._cursor.execute(
                "CREATE TABLE IF NOT EXISTS output_data (key TEXT, value TEXT)"
            )
            self._cursor.execute(
                "CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, status TEXT)"
            )
            self._connection.commit()
        except sqlite3.Error as e:
            print(f"Error connecting to database or creating table: {e}")
            raise

    def __enter__(self) -> None:
        return self

    def __exit__(self, ext_type, exc_value, traceback) -> None:
        """Close database connection before destroying the database instance

        Args:
            this method should not be called directly"""
        self.cursor.close()
        if isinstance(exc_value, Exception):
            self._connection.rollback()
        else:
            self._connection.commit()
        self._connection.close()

    def insert_value_dicts(self, value_dicts: Sequence[Mapping[str, Any]]) -> None:
        """insert key value pairs in value_dicts into database

        Args:
            value_dicts (Sequence[Mapping[str, Any]]): list of value dicts."""
        with self._lock:
            for value_dict in value_dicts:
                for item in value_dict.items():
                    self._cursor.execute(
                        "INSERT INTO output_data VALUES (?, ?)", (item[0], item[1])
                    )
            self._connection.commit()
    
    def insert_job(self, job_id: int, status: str) -> None:
        """update status of the job

        Args:
            job_id (int): the job id of the flow run
            status (str): the status of the job
        """
        with self._lock:
            self._cursor.execute("INSERT INTO jobs VALUES (?, ?)", (job_id, status))
            self._connection.commit()
    
    def update_job(self, job_id: int, status: str) -> None:
        """update status of the job

        Args:
            job_id (int): the job id of the flow run
            status (str): the status of the job
        """
        with self._lock:
            self._cursor.execute("UPDATE job SET status = ? WHERE job_id = ?", (status, job_id))
            self._connection.commit()
    
    def get_job_status(self, job_id: int) -> str:
        with self._lock:
            self._cursor.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id))
            return self._cursor.fetchone()

    def select_all(self, table_name: Optional[str]='output_data', limit: Optional[int]=0, offset: Optional[int]=0) -> Sequence[str]:
        """get all records in the database
        
        Args:
            table_name (str): table name, including 'job' and 'output_data'. Default value is 'output_data'
            limit (Optional[int]): query limit. Default value is 0
            offset (Optional[int]): query offset. Default value is 0

        Returns:
            Sequence[str]: all key value pairs in the database"""
        with self._lock:
            self._cursor.execute("SELECT * FROM ? LIMIT ? OFFSET ?", (table_name, limit, offset))
            return self._cursor.fetchall()
        
    def count_all(self, table_name: Optional[str]='output_data') -> Sequence[str]:
        """count all records in the database
        
        Args:
            table_name (str): table name, including 'job' and 'output_data'. Default value is 'output_data'

        Returns:
            Sequence[str]: all key value pairs in the database"""
        with self._lock:
            self._cursor.execute("SELECT COUNT(*) FROM ?", (table_name))
            return self._cursor.fetchone()
