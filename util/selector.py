import importlib

def get_trainer_and_model(condition):
    if condition == "fast":
        trainer_module = importlib.import_module('util.dtls_model.dtls_fast')
        model_module = importlib.import_module('util.dtls_model.dtls_fast')
        trainer = getattr(trainer_module, 'Trainer')
        model = getattr(model_module, 'DTLS')
    else:
        raise ValueError("Invalid condition. Choose 'fast'.")

    return trainer, model

def get_unet_type(condition):
    if condition == "new":
        unet_module = importlib.import_module('util.unet')
        unet = getattr(unet_module, 'Unet')
        print("============= Model: Unet =============")
    else:
        raise ValueError("Invalid condition. Choose 'new'.")

    return unet
