from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model,save_plot_accuracy, save_model, get_prediction, get_log_path, create_log, get_callbacks
import argparse
import logging
import os
import numpy as np
#logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
#log_dir = "logs"
#os.makedirs(log_dir, exist_ok=True)
#logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str,
#filemode="a")

def training(config_path):

    #logging.info("Starting Model training")

    # Reading in the config file that contains all the necessary parameters from config.yaml file
    config = read_config(config_path)
    #logging.info(f"These are the parameters uses for this NN model: {config}")

    # Importing the necessary dynamic paramters from config.yaml file
    validation_datasize = config["params"]["validation_datasize"]

    # Preparing the data
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)

    # Creating the logs dir
    log_dir_name = get_log_path()
    create_log(log_dir_name, X_train)

    # Importing the necessary dynamic paramters from config.yaml file
    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NO_OF_CLASSES = config["params"]["no_of_classes"]

    # Creating an untrained ANN model
    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NO_OF_CLASSES)

    # Importing the necessary dynamic paramters from config.yaml file
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    ckpt_dir = config["artifacts"]["ckpt_dir"]
    ckpt_dir_path = os.path.join(artifacts_dir, ckpt_dir)
    os.makedirs(ckpt_dir_path, exist_ok=True)
    ckpt_name = config["artifacts"]["ckpt_name"]
    ckpt_path_name = os.path.join(ckpt_dir_path, ckpt_name)
    early_stopping_patience = config["params"]["early_stopping_patience"]
    # Creating the necessary callbacks for the model
    callbacks = get_callbacks(log_dir_name, early_stopping_patience, ckpt_path_name)



    EPOCHS = config["params"]["epochs"]
    VALIDATION = (X_valid, y_valid)
    #logging.info("Starting model training")

    # Training the ANN model
    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION, callbacks=callbacks)

    #Saving accuracy plot to artifacts/plots folder
    #artifacts_dir = config["artifacts"]["artifacts_dir"]
    plot_dir = config["artifacts"]["plot_dir"]
    plot_dir_path = os.path.join(artifacts_dir, plot_dir)
    os.makedirs(plot_dir_path, exist_ok=True)
    plot_name = config["artifacts"]["plot_name"]
    save_plot_accuracy(plot_name, history, plot_dir_path)
    #logging.info("Model training complete")

    # Saving trained model to artifacts/model folder:-
    model_dir = config["artifacts"]["model_dir"]
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    model_name = config["artifacts"]["model_name"]
    save_model(model, model_name, model_dir_path)

    # Testing/Evaluating the model on test data
    test_loss, test_accuracy =  model.evaluate(X_test, y_test)
    print("_________________________________________________________________")
    print(f"Test accuracy:- {test_accuracy}")

    print("\n=================================================================")
    print("Training Complete and model is ready to be used for prediction on real world data")

    # Sample prediction
    pred_result = get_prediction(X_test, y_test, model)
    print(f"Prediction result:- {pred_result}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config","-c", default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path = parsed_args.config)
    