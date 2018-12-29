import logging
import numpy as np
from random import random
from ..trainers.lstm_trainer import LSTMModel

try:
    from .base_predictor import BasePredictor
except:
    from base_predictor import BasePredictor



logger = logging.getLogger(__name__)



class LSTMPredictor(BasePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = None
        self.use_days = None
        self.train_data_num = None
        self.model_index = 0
        self.batch = 0
        self.epoch = 0
        self.model_params = {
            "input_size": 4,
            "output_size": 4,
            "hidden_size": 20,
            "num_layers": 3,
            "fc_1": 2 ** 8,
            "dropout": 0,
        }
        self.lr = 0.0001

    @classmethod
    def init_model_list(cls):
        cls.mode_list = [
            "normal",
        ]
        

    @staticmethod
    def add_param_parser(subparser):
        subparser.add_argument("-u", "--use-days", type=int, required=True, help="use how many days to predict next one")
        subparser.add_argument("-i", "--model-index", type=int, default=0, help="model index")
        subparser.add_argument("--train-data", type=int, required=True, help="produce how many data as training set")
        subparser.add_argument("--batch", type=int, default=10, help="batch size for training")
        subparser.add_argument("--epoch", type=int, default=10, help="expected training epoches")
        subparser.add_argument("--mean", type=int, default=140, help="expected value of stock")


    def set_trainer_parameters(self, args):
        self.args = args
        self.use_days = args.use_days
        self.train_data_num = args.train_data
        self.model_index = args.model_index
        self.batch = args.batch
        self.epoch = args.epoch
        self.mean = args.mean
        

    def data_preprocess(self, data):
        data = [[item["open_p"], item["closing_p"], item["day_low"], item["day_high"]] for item in data]
        data = np.array(data)
        logger.debug(data.shape)

        x_data = np.empty((self.train_data_num , self.use_days, 4))
        y_data = np.empty((self.train_data_num, 4))
        for i in range(self.train_data_num):
            start_i = random.randint(0, data.shape[0] - self.use_days - 1)
            x_data[i] = np.reshape(data[start_i:start_i + self.use_days, :], (-1, self.use_days, 4))
            y_data[i] = data[start_i + self.use_days]
        
        logger.debug(x_data.shape, y_data.shape)
        
        return (x_data, y_data)


    def predict(self, data):
        pass