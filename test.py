import yaml

def load_config(yaml_file):
    with open(yaml_file) as file:
        config = yaml.safe_load(file)
        
    return config


config = load_config("test_yaml.yaml")
print(config['loss']['Params'].keys())
