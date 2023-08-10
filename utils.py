import yaml


def read_yaml(file_path):

    with open(file_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    return data_loaded

print(read_yaml("config.yml")['PROMPT'].format('aa', 'asfsd'))