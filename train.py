import sys

from argparse import ArgumentParser
from data_controller.data_fetch import MongoFetch
from trainers import (
    SimpleTrainer,
)



TRAINER_TABLE = {
    "simple": SimpleTrainer,
}



if __name__ == "__main__":
    arg_model = sys.argv[sys.argv.index("--model") + 1]


    ### Parser
    parser = ArgumentParser()
    parser.add_argument("--model", action="store", required=True, choices=TRAINER_TABLE.keys(), help="the name of the model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fetch", type=str, action="store", nargs=3, help="fetch data via mongo DB: [ticker] [start_date] [end_date]")
    group.add_argument("--download", type=str, action="store", nargs=4, help="fetch data via mongo DB and download: [ticker] [start_date] [end_date] [save file path]")
    group.add_argument("--load", action="store", help="load data via local data: [file path]")
    
    Trainer = TRAINER_TABLE[arg_model]
    subparsers = parser.add_subparsers()
    
    trainer_param_subparser = subparsers.add_parser("param")
    Trainer.add_param_parser(trainer_param_subparser)

    args = parser.parse_args()
    Trainer.set_trainer_parameters(args)


    ### Data loading / fetching
    mongo_fetch = MongoFetch()
    if args.fetch:
        data = mongo_fetch.fetch(*args.fetch)
    elif args.download:
        data = mongo_fetch.fetch(*args.download[:3])
        mongo_fetch.dump_to_np(args.download[-1], data)
    elif args.load:
        data = mongo_fetch.load_from_np(args.load)


    ### Training
    trainer = Trainer()
    preprocessed_data = trainer.data_preprocess(data)
    trainer.load_data(preprocessed_data)
    trainer.train()