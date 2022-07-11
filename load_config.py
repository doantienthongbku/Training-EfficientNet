import yaml
import torch.nn as nn

def load_config(yaml_file):
    with open(yaml_file) as file:
        config = yaml.safe_load(file)
        
    return config

def load_model(config):
    model_choose = {
        'B0': "efficientnet-b0",
        'B1': "efficientnet-b1",
        'B2': "efficientnet-b2",
        'B3': "efficientnet-b3",
        'B4': "efficientnet-b4",
        'B5': "efficientnet-b5",
        'B6': "efficientnet-b6",
        'B7': "efficientnet-b7",
        'B8': "efficientnet-b8"
    }
    
    setting = config['model']
    
    # import model
    cmd_import = f"from {setting['module']} import {setting['class']}"
    exec(cmd_import)
    
    # choose mode of model and set parameters
    if setting['Params']['pretrained']:
        cmd_load_model = f"{setting['class']}.from_pretrained('{model_choose[setting['Params']['model']]}',\
            advprop={setting['Params']['advprop']},in_channels={setting['Params']['in_channels']},\
            num_classes={setting['Params']['num_classes']})"
        model = eval(cmd_load_model)
        
        model._fc.out_features = setting['Params']['num_classes']  # change num_classes of model
    else:
        cmd_load_model = f"{setting['class']}.from_name({model_choose(setting['Params']['model'])}), in_channels={setting['Params']['in_channels']})"
        model = eval(cmd_load_model)
        
        model._fc.out_features = setting['Params']['num_classes']  # change num_classes of model

    return model



def load_engine(config, engine):
    assert engine in ['loss', 'optim', 'lr_scheduler', 'early_stopping']
    
    setting = config[engine]
    
    # import loss
    cmd_import = f"from {setting['module']} import {setting['class']}"
    exec(cmd_import)
    
    cmd_optim = f"{setting['class']}("
    for key, val in setting['Params'].items():
        cmd_params = f"{key}={val},"
        cmd_optim += cmd_params
        
    cmd_optim += ")"
    
    optim = eval(cmd_optim)
    return optim


if __name__ == '__main__':
    config = load_config("config/training_config.yaml")
    model = load_model(config)
    optim = load_optim(config)
    print(optim)
    