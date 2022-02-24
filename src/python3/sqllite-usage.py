import sqlite3

conn = sqlite3.connect("xx.db")
c = conn.cursor()
cursor = conn.execute("select * from xxx")
count = 0
for row in cursor:
    print(row)
    count += 1
print(count)
