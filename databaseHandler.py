from pysondb import db

a=db.getDb("path/to/json.json")
a.addMany([{"name":"pysondb","type":"DB"},{"name":"pysondb-cli","type":"CLI"}])
a.getAll()
[{"name":"pysondb","type":"DB"},{"name":"pysondb-cli","type":"CLI"}]