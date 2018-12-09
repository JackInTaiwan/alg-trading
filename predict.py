import sys
import logging
import numpy as np

from argparse import ArgumentParser
from predictors import SimplePredictor
from data_controller.data_fetch import MongoFetch



logging.basicConfig(level=logging.DEBUG, format="[%(levelname)-0s] %(name)-0s >> %(message)-0s")

PREDICT_MODE = {
    "normal": "Use previous n days data to predict next one where output is a 1x4 np.array containing open, close, high and low prices in form of number.",
    "suggest": "Use previous n days data to predict next one where output is a 1x5 np.array denoting suggetion intention.",
}
PREDICTOR_TABLE = {
    "simple": SimplePredictor,
}


if __name__ == "__main__":
    arg_predictor = sys.argv[sys.argv.index("--predictor") + 1]

    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, action="store", required=True, choices=PREDICT_MODE.keys(), help="select prediction mode")
    parser.add_argument("--predictor", type=str, action="store", required=True, choices=PREDICTOR_TABLE.keys(), help="select predictor")
    parser.add_argument("-s", "--start_date", type=str, action="store", required=True, help="the start date you want to predict: [yyymmdd]")
    parser.add_argument("-e", "--end_date", type=str, action="store", required=True, help="the end date you want to predict: [yyymmdd]")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fetch", type=str, action="store", nargs=3, help="fetch data via mongo DB: [ticker] [start_date] [end_date]")
    group.add_argument("--download", type=str, action="store", nargs=4, help="fetch data via mongo DB and download: [ticker] [start_date] [end_date] [save file path]")
    group.add_argument("--load", action="store", help="load data via local data: [file path]")
    
    Predictor = PREDICTOR_TABLE[arg_predictor]
    Predictor.init_model_list()

    predictor = Predictor()
    subparsers = parser.add_subparsers()
    predictor_param_subparser = subparsers.add_parser("param")
    predictor.add_param_parser(predictor_param_subparser)
    
    args = parser.parse_args()
    predictor.set_trainer_parameters(args)

    if args.mode not in Predictor.mode_list:
        raise ValueError("The prediction mode '{}' if not supported by the predictor '{}'.".format(args.mode, args.predictor))

    predictor.set_mode(args.mode)

    mongo_fetch = MongoFetch()
    if args.fetch:
        data = mongo_fetch.fetch(*args.fetch)
    elif args.download:
        data = mongo_fetch.fetch(*args.download[:3])
        mongo_fetch.dump_to_np(args.download[-1], data)
    elif args.load:
        data = mongo_fetch.load_from_np(args.load)
        
    preprocessed_data = predictor.data_preprocess(data)
    prediction = predictor.predict(preprocessed_data)