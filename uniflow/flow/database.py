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
                    print("new database")
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, filename: str='uniflow_data.db') -> None:
        """Initialize Database class.
        
        Args: 
            filename (str): the filename of the database"""
        print("Initializing database")
        self.filename = filename
        with self as db:
            db._create_tables()
        
    def __enter__(self) -> None:
        """Connect to database."""
        self._lock.acquire()
        self._connection = sqlite3.connect(self.filename)
        self._cursor = self._connection.cursor()
        
    def __exit__(self, ext_type, exc_value, traceback) -> None:
        """Close database connection before destroying the database instance"""
        self._cursor.close()
        if isinstance(exc_value, Exception):
            self._connection.rollback()
        else:
            self._connection.commit()
        self._connection.close()
        self._lock.release()
    
    def _create_tables(self) -> None:
        """Create and connect to database."""
        self._cursor.execute(
            "CREATE TABLE IF NOT EXISTS output_data (key TEXT, value TEXT)"
        )
        self._cursor.execute(
            "CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, status TEXT)"
        )

    def insert_value_dicts(self, value_dicts: Sequence[Mapping[str, Any]]) -> None:
        """insert key value pairs in value_dicts into database

        Args:
            value_dicts (Sequence[Mapping[str, Any]]): list of value dicts."""
        for value_dict in value_dicts:
            for item in value_dict.items():
                self._cursor.execute(
                    "INSERT INTO output_data VALUES (?, ?)", (item[0], item[1])
                )
    
    def insert_job(self, job_id: int, status: str) -> None:
        """update status of the job

        Args:
            job_id (int): the job id of the flow run
            status (str): the status of the job
        """
        self._cursor.execute("INSERT INTO jobs VALUES (?, ?)", (job_id, status))
    
    def update_job(self, job_id: int, status: str) -> None:
        """update status of the job

        Args:
            job_id (int): the job id of the flow run
            status (str): the status of the job
        """
        self._cursor.execute("UPDATE jobs SET status = ? WHERE job_id = ?", (status, job_id))
    
    def get_job_status(self, job_id: int) -> str:
        self._cursor.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,))
        status = self._cursor.fetchone()[0]
        return status

    def select_all(self, table_name: Optional[str]='output_data', limit: Optional[int]=0, offset: Optional[int]=0) -> Sequence[str]:
        """get all records in the database
        
        Args:
            table_name (str): table name, including 'jobs' and 'output_data'. Default value is 'output_data'
            limit (Optional[int]): query limit. Default value is 0
            offset (Optional[int]): query offset. Default value is 0

        Returns:
            Sequence[str]: all key value pairs in the database"""
        if table_name not in ['jobs', 'output_data']:
            raise ValueError("Invalid table name")
        self._cursor.execute(f"SELECT * FROM {table_name} LIMIT ? OFFSET ?", (limit, offset))
        result = self._cursor.fetchall()
        return result
        
    def count_all(self, table_name: Optional[str]='output_data') -> int:
        """count all records in the database
        
        Args:
            table_name (str): table name, including 'jobs' and 'output_data'. Default value is 'output_data'

        Returns:
            Sequence[str]: all key value pairs in the database"""
        if table_name not in ['jobs', 'output_data']:
            raise ValueError("Invalid table name")
        self._cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        result = self._cursor.fetchone()[0]
        return result
