import sqlite3

# Connect to the database
conn = sqlite3.connect('../uniflow_data.db')
cursor = conn.cursor()

# Execute the query to retrieve all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# For each table, print the table name and all its records
for table in tables:
    print(f"Table: {table[0]}")
    cursor.execute(f"SELECT * FROM {table[0]}")
    records = cursor.fetchall()
    for record in records:
        print(record)
        
    cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
    count = cursor.fetchone()
    print(count)

# Close the connection
conn.close()