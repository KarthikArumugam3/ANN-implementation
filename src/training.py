from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model,save_plot_accuracy, save_model
import argparse
import logging
import os

#logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
#log_dir = "logs"
#os.makedirs(log_dir, exist_ok=True)
#logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str,
#filemode="a")

def training(config_path):

    #logging.info("Starting Model training")
    config = read_config(config_path)
    #logging.info(f"These are the parameters uses for this NN model: {config}")

    validation_datasize = config["params"]["validation_datasize"]

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)

    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NO_OF_CLASSES = config["params"]["no_of_classes"]

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NO_OF_CLASSES)

    EPOCHS = config["params"]["epochs"]
    VALIDATION = (X_valid, y_valid)
    #logging.info("Starting model training")
    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION)

    #Saving accuracy plot to artifacts folder
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    plot_dir = config["artifacts"]["plot_dir"]
    plot_dir_path = os.path.join(artifacts_dir, plot_dir)
    os.makedirs(plot_dir_path, exist_ok=True)
    plot_name = config["artifacts"]["plot_name"]
    save_plot_accuracy(plot_name, history, plot_dir_path)
    #logging.info("Model training complete")

    # Saving trained model to artifacts folder:-
    model_dir = config["artifacts"]["model_dir"]
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    model_name = config["artifacts"]["model_name"]
    save_model(model, model_name, model_dir_path)

   

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config","-c", default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path = parsed_args.config)
    