import sqlite3

conn = sqlite3.connect("db1.db")
data = conn.execute("SELECT * FROM user")
for i in data:
    print(i)
d = input("enter the number ")
conn.execute("DELETE FROM user where id = 1"+ d)
conn.commit()
data = conn.execute("SELECT * FROM user")
for i in data:
    print(i)
conn.close()