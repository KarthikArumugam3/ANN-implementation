from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model
import argparse

def training(config_path):
    config = read_config(config_path)

    validation_datasize = config["params"]["validation_datasize"]

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)

    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NO_OF_CLASSES = config["params"]["no_of_classes"]

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NO_OF_CLASSES)

    EPOCHS = config["params"]["epochs"]
    VALIDATION = (X_valid, y_valid)
    
    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION)
   

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config","-c", default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path = parsed_args.config)
    