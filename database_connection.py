import sqlite3
from datetime import datetime

## Connection
conn = sqlite3.connect('HidrolatinaDataBase.db')
c = conn.cursor()

##Create tables
sql = 'create table if not exists ' + 'det_limit' + ' ([id] INTEGER PRIMARY KEY AUTOINCREMENT, [number] INTEGER NOT NULL, [created_at] DATETIME NOT NULL, [update_at] DATETIME NOT NULL)'
c.execute(sql)

sql = 'create table if not exists ' + 'threshold' + ' ([id] INTEGER PRIMARY KEY AUTOINCREMENT, [number] INTEGER NOT NULL, [created_at] DATETIME NOT NULL, [update_at] DATETIME NOT NULL)'
c.execute(sql)

sql = 'create table if not exists ' + 'iou_threshold' + ' ([id] INTEGER PRIMARY KEY AUTOINCREMENT, [number] INTEGER NOT NULL, [created_at] DATETIME NOT NULL, [update_at] DATETIME NOT NULL)'
c.execute(sql)

sql = 'create table if not exists ' + 'user' + ' ([id] INTEGER PRIMARY KEY AUTOINCREMENT, [first_name] TEXT NOT NULL, [last_name] TEXT NOT NULL, [identification] TEXT, [created_at] DATETIME NOT NULL, [update_at] DATETIME NOT NULL)'
c.execute(sql)

##Insert into tables
c.execute('INSERT INTO testdate VALUES(?)',(datetime.now(),))

c.execute('INSERT INTO det_limit(number,created_at,update_at) VALUES(?,?,?)',[1,datetime.now(),datetime.now()])
# sql = 'insert into ' + 'det_limit' + ' (number) values (%d)' % (1)
# c.execute(sql)

c.execute('INSERT INTO threshold(number,created_at,update_at) VALUES(?,?,?)',[1,datetime.now(),datetime.now()])
# sql = 'insert into ' + 'threshold' + ' (number) values (%d)' % (1)
# c.execute(sql)

c.execute('INSERT INTO iou_threshold(number,created_at,update_at) VALUES(?,?,?)',[1,datetime.now(),datetime.now()])
# sql = 'insert into ' + 'iou_threshold' + ' (number) values (%d)' % (1)
# c.execute(sql)

c.execute('INSERT INTO user(first_name,last_name,identification,created_at,update_at) VALUES(?,?,?,?,?)',['nombre','apellido','db7bXdv2vd',datetime.now(),datetime.now()])

##Saves changes
conn.commit()

##Close connection
conn.close()

# c.execute('''CREATE TABLE CLIENTS
#              ([generated_id] INTEGER PRIMARY KEY,[Client_Name] text, [Country_ID] integer, [Date] date)''')

# c.execute('''CREATE TABLE COUNTRY
#              ([generated_id] INTEGER PRIMARY KEY,[Country_ID] integer, [Country_Name] text)''')
        
# c.execute('''CREATE TABLE DAILY_STATUS
#              ([Client_Name] text, [Country_Name] text, [Date] date)''')

# try:
#     sqliteConnection = sqlite3.connect('SQLite_Python.db')
#     sqlite_create_table_query = '''CREATE TABLE SqliteDb_developers (
#                                 id INTEGER PRIMARY KEY,
#                                 name TEXT NOT NULL,
#                                 email text NOT NULL UNIQUE,
#                                 joining_date datetime,
#                                 salary REAL NOT NULL);'''

#     cursor = sqliteConnection.cursor()
#     print("Successfully Connected to SQLite")
#     cursor.execute(sqlite_create_table_query)
#     sqliteConnection.commit()
#     print("SQLite table created")

#     cursor.close()

# except sqlite3.Error as error:
#     print("Error while creating a sqlite table", error)
# finally:
#     if sqliteConnection:
#         sqliteConnection.close()
#         print("sqlite connection is closed")

# class DBConnection:

#     # def __init__(self, db_name='hidrolatina.db'):
#     #     self.name = db_name
#     #     # connect takes url, dbname, user-id, password
#     #     self.conn = sqlite3.connect('hidrolatina.db')
#     #     self.cursor = self.conn.cursor()
#     #     print('funciona?')

#     instance = None
#     def __new__(cls, *args, **kwargs):
#         if cls.instance is None:
#             cls.instance = super().__new__(DBConnection)
#             return cls.instance
#         return cls.instance

#     def __init__(self, db_name='hidrolatina'):
        # self.name = db_name
#         # connect takes url, dbname, user-id, password
#         self.conn = self.connect(db_name)
#         self.cursor = self.conn.cursor()

#     def connect(self):
#         print('funcuona?')
#         try:
#             return sqlite3.connect(self.name)
#         except sqlite3.Error as e:
#             pass

#     def __del__(self):
#         self.cursor.close()
#         self.conn.close()