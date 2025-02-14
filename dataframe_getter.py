import pickle

import mysql.connector
import pandas as pd

from connection_data import HOST, USER, PASSWORD, DATABASE

def load_from_db(table_name):
    global cursor, results
    connection = mysql.connector.connect(
        host=HOST,
        user=USER,
        password=PASSWORD,
        database=DATABASE
    )
    query = "SELECT * FROM " + table_name
    cursor = connection.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    results = pd.DataFrame.from_records(rows, columns=[x[0] for x in cursor.description])
    connection.close()
    filehandler = open("df_" + table_name + ".bin", "wb")
    pickle.dump(results, filehandler)
    filehandler.close()
    return results


def getDataframe(table_name, force_reload=False):
    global cursor, results
    if not force_reload:
        try:
            file = open("df_"+table_name+".bin", 'rb')
            results = pickle.load(file)
            file.close()
            print("INFO: Loading log data from file. Please delete df_"+table_name+".bin if you need to start over.")
        except FileNotFoundError:
            results = load_from_db(table_name)
    else:
        results = load_from_db(table_name)
    return results
