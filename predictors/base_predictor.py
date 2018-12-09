import abc



class BasePredictor(metaclass=abc.ABCMeta):
    mode_list = []


    def __init__(self):
        self.data = None
        self.mode = None


    @classmethod
    @abc.abstractmethod
    def init_model_list(cls):
        """
            cls.model_list = your_model_list
        """
        raise NotImplementedError


    @staticmethod
    @abc.abstractmethod
    def add_param_parser(subparser):
        raise NotImplementedErro


    @staticmethod
    @abc.abstractmethod
    def set_trainer_parameters(args):
        raise NotImplementedError
        

    def set_mode(self, mode):
        self.mode = mode


    @abc.abstractmethod
    def data_preprocess(self, data):
        """
        return: data (any type you want)
        """
        raise NotImplementedError


    def load_data(self, data):
        self.__data = data


    @abc.abstractmethod
    def predict(self, data):
        """
        return: predicted_data [np.array]
        """
        raise NotImplementedError