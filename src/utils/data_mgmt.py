import tensorflow as tf

import matplotlib.pyplot as plt
import os
import pandas as pd

def get_data(validation_datasize):

    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    # data split 
    # # And also normalising them to be between 0&1 by dividing by 255
    # # So out of 60k first 5k will be validation and rest 55k will be used for our training purposes
    X_valid, X_train = X_train_full[:validation_datasize]/255., X_train_full[validation_datasize:]/255.
    y_valid, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:]

    #Scale the test data as well
    X_test = X_test/255.


    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def plot_accuracy(file_name, model,plot_dir_name):

    pd.DataFrame(model.history).plot(figsize=(10,7))
    plt.grid(True)


    plot_dir = plot_dir_name
    os.makedirs(plot_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
    plotPath = os.path.join(plot_dir, file_name) # model/filename
    plt.savefig(plotPath) 

