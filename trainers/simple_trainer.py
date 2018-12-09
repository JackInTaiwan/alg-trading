import logging
import numpy as np

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor as MLPR

try:
    from .base_trainer import BaseTrainer
except:
    from base_trainer import BaseTrainer



logger = logging.getLogger(__name__)
random = np.random.RandomState(0)


class SimpleTrainer(BaseTrainer):
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

        x_data = np.empty((self.train_data_num , 4 * self.use_days))
        y_data = np.empty((self.train_data_num, 4))
        for i in range(self.train_data_num):
            start_i = random.randint(0, data.shape[0] - self.use_days - 1)
            x_data[i] = np.reshape(data[start_i:start_i + self.use_days, :], (-1, 4 * self.use_days))
            y_data[i] = data[start_i + self.use_days]
        
        logger.debug(x_data.shape)
        logger.debug(y_data)
        
        return (x_data, y_data)


    def train(self):
        x_data, y_data = self.data
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
        
        logger.debug((x_train.shape, x_test.shape, y_train.shape, y_test.shape))

        hidden_layer_sizes = (self.use_days * 4, 50, 4)
        mlpr = MLPR(hidden_layer_sizes=hidden_layer_sizes, activation='logistic',
            solver='lbfgs', batch_size='auto',
            learning_rate='constant', learning_rate_init=10 ** -9,
            max_iter=500, shuffle=True
        )
        mlpr.fit(x_train, y_train)

        pred = mlpr.predict(x_test)
        for x, y in zip(pred, y_test):
            logger.debug((x, y))

        self.save(mlpr)


    def save(self, model):
        import os
        
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "simple",
            "simple_{}.pkl".format(self.model_index)
        )

        joblib.dump(model, path)