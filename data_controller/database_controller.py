from pymongo import MongoClient, ASCENDING



def preprocess_data(fp, fn):
    with open(fp, "r") as f:
        data = f.read()

    data = "[" + data.replace("]][[", "]],[[") + "]"
    data = eval(data)

    data_flatten = []

    for month_item in data:
        for day_item in month_item:
            data_flatten.append(day_item)

    check_abundant(data_flatten)
    check_attr(data_flatten)

    data_db = parse_data(data_flatten, fn)

    return data_db



def check_abundant(data):
    for item in data:
        if len(item) != 9:
            raise ValueError("One item is not abundant which lacks some attributes. In file {}:{}".format(fp, item))

    return None



def check_attr(data):
    for item in data:
        pattern = "/"
        count_pattern = [True for attr in item if pattern in attr].count(True)
        if count_pattern != 1 or pattern not in item[0]:
            raise ValueError("One item contains wrong number of '/'. In file {}:{}".format(fp, item))

        # else attribute checking can be append here

    return None



def parse_data(data, fn):
    # ["y", "m", "d", "trading_volume", "turnover_in_value", "open_price", "day_high", "day_low", "closing_price", "fluctuation_difference", "number_of_transactions"]
    attrs = ["y", "m", "d", "t_vol", "t_in_val", "open_p", "day_high", "day_low", "closing_p", "fluct_diff", "n_of_trans"]

    for i, item in enumerate(data):
        try:
            new_item = dict()
            item = [int(value) for value in item[0].split("/")] + item[1:]
            for attr, value in zip(attrs, item):
                new_item[attr] = value
            # add index for database
            new_item["ticker"] = int(fn)
            new_item["u_id"] = int("{:0>4}{:0>3}{:0>2}{:0>2}".format(fn, int(new_item["y"]), int(new_item["m"]), int(new_item["d"])))
            data[i] = new_item

        except Exception as e:
            print("Data missing some values:")
            print("File: {}".format(fn))
            print(item)

    data = sorted(data, key=lambda x: x["u_id"])

    tmp = None
    u_id_tmp = None
    head_missing_catch = []

    for i, item in enumerate(data):
        try:
            if item["u_id"] != u_id_tmp:
                missing_data_toggle = False
                for attr, value in item.items():
                    if type(value) is str:
                        patterns = [",", "X"]
                        patterns_lack = "--"
                        for pattern in patterns: value = value.replace(pattern, "")
                        if patterns_lack in value:
                            item[attr] = tmp[attr]
                            missing_data_toggle = True
                        else :
                            item[attr] = float(value)
                if not missing_data_toggle: tmp = item
                u_id_tmp = item["u_id"]
            else:
                head_missing_catch.append(i)
        except Exception as e:
            head_missing_catch.append(i)

    for index in reversed(head_missing_catch):
        data.pop(index)

    return data



def save_data_to_db(data, db_name, db_acc, db_pwd):
    fp = "./data/8341"
    # preprocess_data(fp)
    db_url = "mongodb://{}:{}@ds121343.mlab.com:21343/alg-trading-2".format(db_acc, db_pwd)
    conn = MongoClient(db_url)
    db = conn[db_name]
    col = db["stock-history"]
    col.create_index([("u_id", ASCENDING),], unique=True)
    col.insert(data)



if __name__ == "__main__":
    import os
    import sys

    index_start = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    dir = "./data"
    for fn in sorted(os.listdir(dir))[index_start:]:
        print("Process {} now...".format(fn))
        fp = os.path.join(dir, fn)
        db_name = "alg-trading-2"
        db_acc = "admin"
        db_pwd = "admin0101"
        data_db =  preprocess_data(fp, fn)
        print("Saving {} now...".format(fn))
        print("Total {} docs.\n".format(len(data_db)))
        save_data_to_db(data_db, db_name, db_acc, db_pwd)

    # print(len(os.listdir(dir)))
    # print(os.listdir(dir).index("1312"))
    # fp = "./data/1451"
    # data_db = preprocess_data(fp, fn)
