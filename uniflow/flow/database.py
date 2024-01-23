"""Database class."""
import logging
from typing import Optional, Sequence, Mapping, Any
import sqlite3
from threading import Lock

logger = logging.getLogger(__name__)


class Database:
    """Database class. Singleton. Thread-safe."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Singleton: create only 1 Database instance globally in the process
        and return the same one when called again.

        Args:
            cls (Database): the Database class"""
        if cls._instance is None:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, filename: str = "uniflow_data.db") -> None:
        """Initialize Database class.

        Args:
            filename (str): the filename of the database"""
        self.filename = filename
        self._create_tables_if_not_exist()

    def __enter__(self) -> None:
        """Connect to database."""
        if not self._lock.locked():
            raise Exception(
                "Lock is not acquired. The __enter__ function needs to be locked and _lock shall be private. Therefore, the with statement should only be used in the member functions of Database class"
            )
        self._connection = sqlite3.connect(self.filename)
        self._cursor = self._connection.cursor()
        return self

    def __exit__(self, ext_type, exc_value, traceback) -> None:
        """Commit and Close database connection"""
        self._cursor.close()
        if isinstance(exc_value, Exception):
            self._connection.rollback()
        else:
            self._connection.commit()
        self._connection.close()

    def _create_tables_if_not_exist(self) -> None:
        """Create the database table if it does not exist."""
        with self._lock, self:
            self._cursor.execute(
                "CREATE TABLE IF NOT EXISTS output_data (key TEXT PRIMARY KEY, value TEXT)"
            )
            self._cursor.execute(
                "CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, status TEXT)"
            )

    def insert_value_dicts(self, value_dicts: Sequence[Mapping[str, Any]]) -> None:
        """insert key value pairs in value_dicts into database
            If the key already exists, update the value

        Args:
            value_dicts (Sequence[Mapping[str, Any]]): list of value dicts."""
        with self._lock, self:
            for value_dict in value_dicts:
                for item in value_dict.items():
                    self._cursor.execute(
                        "SELECT * FROM output_data WHERE key = ?", (item[0],)
                    )
                    result = self._cursor.fetchone()
                    if result:
                        self._cursor.execute(
                            "UPDATE output_data SET value = ? WHERE key = ?",
                            (item[1], item[0]),
                        )
                    else:
                        self._cursor.execute(
                            "INSERT INTO output_data VALUES (?, ?)", (item[0], item[1])
                        )

    def update_job_status(self, job_id: str, status: str) -> None:
        """Insert or update status of the job

        Args:
            job_id (str): the job id of the flow run
            status (str): the status of the job"""
        with self._lock, self:
            self._cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            result = self._cursor.fetchone()
            if result:
                self._cursor.execute(
                    "UPDATE jobs SET status = ? WHERE job_id = ?", (status, job_id)
                )
            else:
                self._cursor.execute("INSERT INTO jobs VALUES (?, ?)", (job_id, status))

    def get_job_status(self, job_id: str) -> str:
        """Get the status of the job

        Args:
            job_id (str): the job id of the flow run

        Returns:
            str: the status of the job"""
        with self._lock, self:
            self._cursor.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,))
            status = self._cursor.fetchone()
            if status:
                return status[0]
            else:
                return ""

    def select_all(
        self,
        table_name: Optional[str] = "output_data",
        limit: Optional[int] = 0,
        offset: Optional[int] = 0,
    ) -> Sequence[str]:
        """get all records in the database

        Args:
            table_name (str): table name, including 'jobs' and 'output_data'. Default value is 'output_data'
            limit (Optional[int]): query limit. Default value is 0
            offset (Optional[int]): query offset. Default value is 0

        Returns:
            Sequence[str]: all key value pairs in the database"""
        with self._lock, self:
            if table_name not in ["jobs", "output_data"]:
                raise ValueError("Invalid table name")
            self._cursor.execute(
                f"SELECT * FROM {table_name} LIMIT ? OFFSET ?", (limit, offset)
            )
            result = self._cursor.fetchall()
            return result

    def count_all(self, table_name: Optional[str] = "output_data") -> int:
        """count all records in the database

        Args:
            table_name (str): table name, including 'jobs' and 'output_data'. Default value is 'output_data'

        Returns:
            int: count of all records in the database"""
        with self._lock, self:
            if table_name not in ["jobs", "output_data"]:
                raise ValueError("Invalid table name")
            self._cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            result = self._cursor.fetchone()[0]
            return result
