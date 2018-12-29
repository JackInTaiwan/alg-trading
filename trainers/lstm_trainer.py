import logging
import numpy as np
import torch as tor
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

try:
    from .base_trainer import BaseTrainer
except:
    from base_trainer import BaseTrainer



logger = logging.getLogger(__name__)
random = np.random.RandomState(0)


class LSTMTrainer(BaseTrainer):
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


    def train(self):
        ### Load date
        x_data, y_data = self.data
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
        x_train, x_test, y_train, y_test = x_train - self.mean, x_test - self.mean, y_train - self.mean, y_test - self.mean
        logger.info("data sizes: {}".format((x_train.shape, x_test.shape, y_train.shape, y_test.shape)))
        
        data_set = TensorDataset(tor.FloatTensor(x_train), tor.FloatTensor(y_train))
        data_loader = DataLoader(
            dataset=data_set,
            batch_size=self.batch,
            shuffle=True,
            drop_last=True,
        )
        
        ### Model params
        lstm = LSTMModel(self.model_params)
        loss_func = tor.nn.MSELoss()
        optim = tor.optim.Adam(lstm.parameters(), lr=self.lr)

        ### Training
        for epoch in range(self.epoch):
            for step, (x, y) in enumerate(data_loader):
                optim.zero_grad()
                pred = lstm(x)
                
                loss = loss_func(pred, y)
                logger.info("Loss: {}".format(loss))

                loss.backward()
                optim.step()

            optim.zero_grad()
            y = tor.FloatTensor(y_test[:10])
            pred = lstm(tor.FloatTensor(x_test[:10]))
            logger.debug(pred)
            # logger.debug(y)
            loss_valid = loss_func(pred, y)
            logger.info("Loss on valid: {}".format(loss_valid))

            if epoch % 50 == 0:
                self.save(lstm, self.model_params, self.args)


    def save(self, model, model_param, args):
        import os
        import json
        from datetime import datetime

        ### Save model
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "lstm",
            "lstm_{}.pkl".format(self.model_index)
        )

        tor.save(model.state_dict(), path)

        ### Save model information
        with open(path.replace(".pkl", ".json"), "w") as f:
            info = {
                "time": str(datetime.now()),
                "model_param": model_param,
                "args": vars(args),
            }
            json.dump(info, f)
            


class LSTMModel(nn.Module):
    def __init__(self, kwargs):
        super(LSTMModel, self).__init__()

        self.fc_1 = nn.Linear(kwargs["hidden_size"], kwargs["fc_1"])
        self.fc_2 = nn.Linear(kwargs["fc_1"], kwargs["output_size"])
        self.lstm = nn.LSTM(
            input_size=kwargs["input_size"],
            hidden_size=kwargs["hidden_size"],
            num_layers=kwargs["num_layers"],
            dropout=kwargs["dropout"],
            batch_first=True,
        )
        self.sigmoid = tor.nn.Sigmoid()


    def forward(self, x) :
        o, (h_n, c_n) = self.lstm(x)     #(batch, seq, hidden_size), (layer, batch, hidden_size), (layer, batch, hidden_size)
        o = o[:, -1]
        
        o = self.fc_1(o)
        o = self.sigmoid(o)
        o = self.fc_2(o)

        return o