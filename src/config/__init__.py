import yaml
default_cfg = "/home/kevin2li/ut-gan/src/config/default.yml"

def getConfig(filename=default_cfg):
    with open(filename, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg