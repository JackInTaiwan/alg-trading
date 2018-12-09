try:
    from .base_trainer import BaseTrainer
except:
    from base_trainer import BaseTrainer



class SimpleTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    @staticmethod
    def add_param_parser(subparser):
        subparser.add_argument("-a", type=int)

    @staticmethod
    def set_trainer_parameters(args):
        pass


    def data_preprocess(self, data):
        return data


    def train(self):
        print(self.data)
        pass


    def predict(self):
        pass