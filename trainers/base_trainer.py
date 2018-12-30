import abc
import logging



logger = logging.getLogger(__name__)

class BaseTrainer(metaclass=abc.ABCMeta):
    def __init__(self):
        self.data = None


    @staticmethod
    @abc.abstractmethod
    def add_param_parser(subparser):
        raise NotImplementedError


    @abc.abstractmethod
    def set_trainer_parameters(self, args):
        raise NotImplementedError


    @staticmethod
    @abc.abstractmethod
    def add_predict_param_parser(subparser):
        raise NotImplementedError


    @abc.abstractmethod
    def set_predict_trainer_parameters(self, args):
        raise NotImplementedError
        

    @abc.abstractmethod
    def train_data_preprocess(self, data):
        """
        return: data (any type you want)
        """
        raise NotImplementedError
    

    @abc.abstractmethod
    def predict_data_preprocess(self, data):
        """
        return: data (any type you want)
        """
        raise NotImplementedError


    def load_data(self, data):
        self.data = data


    @abc.abstractmethod
    def train(self):
        raise NotImplementedError


    @abc.abstractmethod
    def predict(self):
        raise NotImplementedError