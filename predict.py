import sys
import logging
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from trainers import LSTMTrainer
from data_controller.data_fetch import MongoFetch



logging.basicConfig(level=logging.DEBUG, format="[%(levelname)-0s] %(name)-0s >> %(message)-0s")
logger = logging.getLogger(__name__)

PREDICT_MODE = {
    "normal": "Use previous n days data to predict next one where output is a 1x4 np.array containing open, close, high and low prices in form of number.",
    "suggest": "Use previous n days data to predict next one where output is a 1x5 np.array denoting suggetion intention.",
}
TRAINER_TABLE = {
    "lstm": LSTMTrainer,
}


def fetch_test_data(ticker, s_date, e_date, use_days):
    from datetime import datetime, timedelta

    s_datetime = datetime(int(s_date[:3]) + 1911, int(s_date[3:5]), int(s_date[5:]))
    s_loose_date = s_datetime - timedelta(days=use_days * 2)
    s_loose_date = "{:0>3}{:0>2}{:0>2}".format(s_loose_date.year - 1911, s_loose_date.month, s_loose_date.day)

    mongo_fetch = MongoFetch()
    data = mongo_fetch.fetch(ticker, s_loose_date, e_date)
    data = np.array(data)
    for i, item in enumerate(data):
        if item["u_id"] > int("{}{}".format(ticker, s_date)): break
    if i < use_days:
        raise ValueError("Parameters 'use-days' is too large that we donnot adequate history data.")
    else:
        data = data[i - use_days:]

    return data


def save_prediction(predictor_name, ticker, s_date, e_date, prediction):
    import os
    if type(prediction) is not np.ndarray:
        prediction = np.array(prediction)
    pd.DataFrame(prediction).to_csv(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "predictions",
            "{}_{}_{}_{}.csv".format(predictor_name, ticker, s_date, e_date)
        )
    )



if __name__ == "__main__":
    arg_trainer = sys.argv[sys.argv.index("--trainer") + 1]

    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, action="store", required=True, choices=PREDICT_MODE.keys(), help="select prediction mode")
    parser.add_argument("--trainer", type=str, action="store", required=True, choices=TRAINER_TABLE.keys(), help="select trainer")
    parser.add_argument("--ticker", type=str, action="store", required=True, help="select ticker")
    parser.add_argument("-u", "--use-days", type=int, required=True, help="use how many days to predict next one")
    parser.add_argument("-s", "--start-date", type=str, action="store", required=True, help="the start date you want to predict: [yyymmdd]")
    parser.add_argument("-e", "--end-date", type=str, action="store", required=True, help="the end date you want to predict: [yyymmdd]")
    
    Trainer = TRAINER_TABLE[arg_trainer]

    trainer = Trainer()
    subparsers = parser.add_subparsers()
    predictor_param_subparser = subparsers.add_parser("param")
    trainer.add_predict_param_parser(predictor_param_subparser)
    
    args = parser.parse_args()
    trainer.set_predict_trainer_parameters(args)

    data = fetch_test_data(args.ticker, args.start_date, args.end_date, args.use_days)
        
    preprocessed_data = trainer.predict_data_preprocess(data)
    trainer.load_data(preprocessed_data)
    prediction = trainer.predict()
    
    save_prediction(arg_trainer, args.ticker, args.start_date, args.end_date, prediction)