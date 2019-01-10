import logging
import numpy as np
import torch as tor
import torch.nn as nn

from torch.autograd import Variable
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
        self.load_path = ""
        self.model_params = {
            "input_size": 4,
            "output_size": 4,
            "hidden_size": 2 ** 9,
            "num_layers": 3,
            "fc_1": 2 ** 9,
            "dropout": 0.5,
        }
        self.lr = 0.0001


    @staticmethod
    def add_param_parser(subparser):
        subparser.add_argument("-i", "--model-index", type=int, default=0, help="model index")
        subparser.add_argument("--train-data", type=int, required=True, help="produce how many data as training set")
        subparser.add_argument("--batch", type=int, default=10, help="batch size for training")
        subparser.add_argument("--epoch", type=int, default=10, help="expected training epoches")
        subparser.add_argument("--gpu", default=False, action="store_true", help="whether use gpu for training")


    @staticmethod
    def add_predict_param_parser(subparser):
        subparser.add_argument("--load", type=str, required=True, help="path of trained model")


    def set_trainer_parameters(self, args):
        self.args = args
        self.use_days = args.use_days
        self.train_data_num = args.train_data
        self.model_index = args.model_index
        self.batch = args.batch
        self.epoch = args.epoch
        self.gpu = args.gpu


    def set_predict_trainer_parameters(self, args):
        self.args = args
        self.use_days = args.use_days
        self.load_path = args.load


    def train_data_preprocess(self, data):
        data = [[item["open_p"], item["closing_p"], item["day_low"], item["day_high"]] for item in data]
        data = np.array(data)

        x_data = np.empty((self.train_data_num , self.use_days, 4))
        y_data = np.empty((self.train_data_num, 4))
        for i in range(self.train_data_num):
            start_i = random.randint(0, data.shape[0] - self.use_days - 1)
            x_data[i] = np.reshape(data[start_i:start_i + self.use_days, :], (-1, self.use_days, 4))
            y_data[i] = data[start_i + self.use_days]
        
        logger.info("preprocessed x_data shape: {}".format(x_data.shape))
        return (x_data, y_data)


    def predict_data_preprocess(self, data):
        data = [[item["u_id"], item["open_p"], item["closing_p"], item["day_low"], item["day_high"]] for item in data]
        data = np.array(data)
        logger.debug("predicted data shape:{}".format(data.shape))

        x_data = np.empty((data.shape[0] - self.use_days , self.use_days, 5))
        date_data = []
        for i in range(x_data.shape[0]):
            x_data[i] = np.reshape(data[i:i + self.use_days, :], (-1, self.use_days, 5))
            date_data.append(data[i + self.use_days][0])
        
        logger.info("preprocessed x_data shape: {}".format(x_data.shape))

        return (date_data, x_data)


    def train(self):
        ### Load date
        x_data, y_data = self.data
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=0)
        x_train, x_test, y_train, y_test = x_train, x_test, y_train, y_test
        logger.info("data sizes: {}".format((x_train.shape, x_test.shape, y_train.shape, y_test.shape)))
        
        data_set = TensorDataset(tor.FloatTensor(x_train), tor.FloatTensor(y_train))
        data_loader = DataLoader(
            dataset=data_set,
            batch_size=self.batch,
            shuffle=True,
            drop_last=True,
        )
        
        ### Model params
        lstm = LSTMModel(self.model_params) if not self.gpu else LSTMModel(self.model_params).cuda()
        loss_func = tor.nn.MSELoss() if not self.gpu else tor.nn.MSELoss().cuda()
        optim = tor.optim.Adam(lstm.parameters(), lr=self.lr)

        ### Training
        for epoch in range(self.epoch):
            total_loss = 0
            for step, (x, y) in enumerate(data_loader):
                x = Variable(x) if not self.gpu else Variable(x).cuda()
                y = Variable(y) if not self.gpu else Variable(y).cuda()
                optim.zero_grad()
                pred = lstm(x)
                
                loss = loss_func(pred, y)
                total_loss += loss
                loss.backward()
                optim.step()
                
            logger.debug("y: {}".format(y))
            logger.debug("mean: {}".format(tor.mean(x, dim=1)))
            logger.debug("pred: {}".format(pred))
            logger.info("total loss:{}".format(total_loss))

            ### Validation loss
            optim.zero_grad()
            y_test = tor.FloatTensor(y_test) if not self.gpu else tor.FloatTensor(y_test).gpu()
            x_test = tor.FloatTensor(x_test) if not self.gpu else tor.Floattensor(x_test).gpu()
            lstm.eval()
            pred = lstm(tor.FloatTensor(x_test))
            lstm.train()

            loss_valid = loss_func(pred, y_test)
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


    def predict(self):
        date_data, x_data = self.data
        x_test = tor.FloatTensor(x_data[:, :, 1:])
        lstm = LSTMModel(self.model_params)
        lstm.load_state_dict(tor.load(self.load_path))
        lstm.eval()

        prediction = np.empty((x_test.shape[0], 5))
        for i, x in enumerate(x_test):
            pred = lstm(x.reshape(1, x.size(0), x.size(1)))
            prediction[i] = np.hstack((np.array(date_data[i]).reshape(1, -1), pred.detach().numpy()))

        return prediction



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
        self.drop = tor.nn.Dropout(p=kwargs["dropout"])


    def forward(self, x) :
        # x: (batch, use_days, 4)
        o, (h_n, c_n) = self.lstm(x)     #(batch, seq, hidden_size), (layer, batch, hidden_size), (layer, batch, hidden_size)
        o = o[:, -1]
        m = tor.mean(x, dim=1)
        o = self.fc_1(o)
        o = self.drop(self.sigmoid(o))
        o = self.fc_2(o)
        o = o + m

        return o
