try:
    from .base_predictor import BasePredictor
except:
    from base_predictor import BasePredictor



class SimplePredictor(BasePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    @classmethod
    def init_model_list(cls):
        cls.mode_list = [
            "normal",
        ]
        

    @staticmethod
    def add_param_parser(subparser):
        subparser.add_argument("--cc")


    def set_trainer_parameters(self, args):
        pass
        

    def data_preprocess(self, data):
        """
        return: data (any type you want)
        """
        return data


    def predict(self, data):
        pass