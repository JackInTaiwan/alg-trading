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
        self.use_days = None
        self.train_data_num = None
        self.model_index = 0


    @staticmethod
    def add_param_parser(subparser):
        subparser.add_argument("-u", "--use-days", type=int, required=True, help="use how many days to predict next one")
        subparser.add_argument("--train-data", type=int, required=True, help="produce how many data as training set")
        subparser.add_argument("-i", "--model-index", type=int, default=0, help="model index")


    def set_trainer_parameters(self, args):
        self.use_days = args.use_days
        self.train_data_num = args.train_data
        self.model_index = args.model_index


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
        
        logger.debug(x_data.shape)
        logger.debug(y_data)
        
        return (x_data, y_data)


    def train(self):
        batch_size = 10
        epoch = 500

        x_data, y_data = self.data
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
        logger.info("data sizes: {}".format((x_train.shape, x_test.shape, y_train.shape, y_test.shape)))
        data_set = TensorDataset(tor.FloatTensor(x_train), tor.FloatTensor(y_train))

        data_loader = DataLoader(
            dataset=data_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        
        lstm = LSTMModel(
            input_size=4,
            output_size=4,
            hidden_size=10,
            num_layers=3,
            dropout=0,
        )

        loss_func = tor.nn.MSELoss()
        optim = tor.optim.Adam(lstm.parameters(), lr=0.01)

        for epoch in range(epoch):
            for step, (x, y) in enumerate(data_loader):
                optim.zero_grad()
                pred = lstm(x)
                
                loss = loss_func(pred, y)
                logger.info("Loss: {}".format(loss))

                loss.backward()
                optim.step()

            logger.info()

    def save(self, model):
        import os
        
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "simple",
            "simple_{}.pkl".format(self.model_index)
        )

        joblib.dump(model, path)



class LSTMModel(nn.Module):
    def __init__(self, **kwargs):
        super(LSTMModel, self).__init__()

        self.fc = nn.Linear(kwargs["hidden_size"], kwargs["output_size"])
        self.lstm = nn.LSTM(
            input_size=kwargs["input_size"],
            hidden_size=kwargs["hidden_size"],
            num_layers=kwargs["num_layers"],
            dropout=kwargs["dropout"],
            batch_first=True,
        )


    def forward(self, x) :
        o, (h_n, c_n) = self.lstm(x)     #(batch, seq, hidden_size), (layer, batch, hidden_size), (layer, batch, hidden_size)
        o = o[:, -1]
        o = self.fc(o)
        return o