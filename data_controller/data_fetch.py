from pymongo import MongoClient
from .utils.schema import Schema

import logging
import numpy as np



schema = Schema()


class MongoFetch:
    def __init__(self, dbUrl=None, dbName=None, dbColName=None):
        self.__dbUrl = dbUrl or "mongodb://admin:admin0101@ds119503.mlab.com:19503/alg-trading"
        self.__dbName = dbName or "alg-trading"
        self.__dbColName = dbColName or "stock-history"
    

    def fetch(self, ticker, start_date, end_date):
        schema.add_schema("ticker", str)
        schema.add_schema("ticker", lambda x: len(x) == 4)
        schema.add_schema("start_date", str)
        schema.add_schema("start_date", lambda x: len(x) == 7)
        schema.add_schema("end_date", str)
        schema.add_schema("end_date", lambda x: len(x) == 7)
        schema.check("ticker", ticker)
        schema.check("start_date", start_date)
        schema.check("end_date", end_date)

        collection = MongoClient(self.__dbUrl)[self.__dbName][self.__dbColName]
        
        data = collection.find({
            "ticker": int(ticker),
            "u_id": {
                "$gte": int("".join([ticker, start_date])),
                "$lte": int("".join([ticker, end_date])),
            }
        })

        data = np.array(list(data))
        return data


    @staticmethod
    def dump_to_np(fp, data):
        np.save(fp, data)


    @staticmethod
    def load_from_np(fp):
        data = np.load(fp)
        return data



if __name__ == "__main__":
    ### Example Code
    mongo_fetch = MongoFetch()
    data = mongo_fetch.fetch("0482", "0990101", "0991210")
    mongo_fetch.dump_to_np(data)
    data = mongo_fetch.load_from_np("./test.npy")