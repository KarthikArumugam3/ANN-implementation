import tensorflow as tf
#import logging
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NO_OF_CLASSES):
   #logging.info("Creating the NN model")
    LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28,28], name="input_layer"),
          tf.keras.layers.Dense(units=300,activation="relu", name="hidden_layer1"),
          tf.keras.layers.Dense(units=100,activation="relu", name="hiddel_layer2"),
          tf.keras.layers.Dense(units=NO_OF_CLASSES,activation="softmax", name="output_layer")
          ]

    # Creating model
    model_clf = tf.keras.models.Sequential(LAYERS)

    model_clf.summary()

   #logging.info(f"Model summary:\n {model_summary}")
   #logging.info("Model created")

    model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)
   #logging.info("Model Compiled")

    return model_clf ## returns an untrained model

def get_unique_plot_name(filename):
    unique_plot_name = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_plot_name

def save_plot_accuracy(plot_name, model,plot_dir):

    #logging.info("Creating the accuracy plot")
    pd.DataFrame(model.history).plot(figsize=(10,7))
    plt.grid(True)
    unique_plotname = get_unique_plot_name(plot_name)
    plotPath = os.path.join(plot_dir, unique_plotname)
    plt.savefig(plotPath)

def get_unique_file_name(filename):
    unique_file_name = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_file_name

def save_model(model, model_name, model_dir):
    unique_filename = get_unique_file_name(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)
