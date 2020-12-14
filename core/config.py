import yaml
from . import config_default


class Dict(dict):
    '''
    Simple dict but support access as x.y style.
    '''
    def __init__(self, names=(), values=(), **kw):
        super(Dict, self).__init__(**kw)
        for k, v in zip(names, values):
            self[k] = v

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(r"'Dict' object has no attribute '%s'" % key)

    def __setattr__(self, key, value):
        self[key] = value


def merge(defaults, override):
    for k, v in defaults.items():
        if k in override:
            if isinstance(v, dict):
                merge(v, override[k])
            else:
                defaults[k] = override[k]
        else:
            defaults[k] = v


def toDict(d):
    D = Dict()
    for k, v in d.items():
        D[k] = toDict(v) if isinstance(v, dict) else v
    return D


configs = config_default.configs
configs = toDict(configs)


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = yaml.load(f, Loader=yaml.FullLoader)
        merge(configs, exp_config)
