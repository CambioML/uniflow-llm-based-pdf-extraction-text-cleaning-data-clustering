"""Database class."""
import logging

import sqlite3
from threading import Lock

logger = logging.getLogger(__name__)


class Database:
    """Database class."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Singleton: only create the Database instance once
        The database instance is protected by the unique _lock

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

    def insert_many(self, value_dicts) -> None:
        """execute a row of data to current cursor

        Args:
            value_dicts (Sequence[Mapping[str, Any]]): list of value dicts.
        """
        with self._lock:
            for value_dict in value_dicts:
                for item in value_dict.items():
                    self._cursor.execute(
                        "INSERT INTO output_data VALUES (?, ?)", (item[0], item[1])
                    )
            self._connection.commit()

    def select_all(self) -> Sequence[str]:
        """execute a row of data to current cursor

        Returns:
            Sequence[str]: all key value pairs in the database"""
        with self._lock:
            self._cursor.execute("SELECT * FROM output_data")
            return self._cursor.fetchall()
