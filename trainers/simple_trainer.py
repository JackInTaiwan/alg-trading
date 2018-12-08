try:
    from .base_trainer import BaseTrainer
except:
    from base_trainer import BaseTrainer



class SimpleTrainer(BaseTrainer):
    def __init__(self, data=None, *args, **kwargs):
        super(SimpleTrainer, self).__init__(*args, **kwargs)
        self.__data = data


    def add_param_parser(subparser):
        subparser.add_argument("-a", type=int)


    def set_trainer_parameters(args):
        pass


    def data_preprocess(self, data):
        return data


    def load_data(self, data):
        self.__data = data


    def train(self):
        pass


    def predict(self):
        pass