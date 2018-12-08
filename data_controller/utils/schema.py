import types



class Schema:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__table = dict()


    def add_schema_dict(self, schema_dict):
        assert(type(schema_dict) is dict)
        for key, item in schema_dict: 
            self.__table[key] = [item,] if key not in self.__table else self.__table[key] + [item, ]


    def add_schema(self, key, item):
        assert(item is not None)
        self.__table[key] = [item,] if key not in self.__table else self.__table[key] + [item, ]


    def check(self, key, var):
        schema_list = self.__table[key]
        for schema in schema_list:
            if isinstance(schema, types.FunctionType):
                if not schema(var):
                    raise ValueError("[Schema Error] The variable '{}' is {} and it violates the schema rule.".format(key, var))
            else:
                if (type(schema) in [list, tuple] and type(var) not in schema) or (type(schema) not in [list, tuple] and type(var) is not schema):
                    raise TypeError("[Schema Error] The variable '{}' has wrong type '{}' while '{}' is expected.".format(key, type(var), schema))