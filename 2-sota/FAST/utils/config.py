import json
from argparse import ArgumentParser
def args2json(args,path):
    with open(path, mode = "w") as f:
        json.dump(args.__dict__, f, indent = 4)

def json2args(config_path):
    import json
    with open(config_path, 'r') as f:
        params = json.load(f)

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    return Namespace(**params)

def load_config(args,path):
    pass
