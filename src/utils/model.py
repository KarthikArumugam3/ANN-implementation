import tensorflow as tf
#import logging
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NO_OF_CLASSES):
   #logging.info("Creating the NN model")
    LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28,28], name="input_layer"),
          tf.keras.layers.Dense(units=300,activation="relu", name="hidden_layer1"),
          tf.keras.layers.Dense(units=100,activation="relu", name="hidden_layer2"),
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

def get_prediction(test_data,test_data_ans,model):
    # Sample prediction function with test data from dataset itself, has to be scaled in future to really unseen data
    print("\n-----------------------------------------------------------------")
    X_test_new = test_data[:3]
    print(f"Sample prediction:- \nActual {test_data_ans[:3]}")
    print("\nModel predicting.....")
    y_pred = np.argmax(model.predict(X_test_new), axis=-1)
    return y_pred

def get_log_path(log_dir="logs/fit"):
  uniqueName = time.strftime("log_%Y_%m_%d_%H_%M_%S")
  log_path = os.path.join(log_dir, uniqueName)
  print(f"savings logs at: {log_path}")

  return log_path

def create_log(log_dir_path,data):
  file_writer = tf.summary.create_file_writer(logdir=log_dir_path)
  with file_writer.as_default():
    images = np.reshape(data[10:30], (-1, 28, 28, 1)) ### <<< 20, 28, 28, 1
    tf.summary.image("20 handwritten digit samples", images, max_outputs=25, step=0)

def get_callbacks(log_dir, early_stopping_patience, ckpt_path_name):
  callbacks_list = []

  # Tensorboard callback
  tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  callbacks_list.append(tensorboard_cb)

  # Early stopping callback
  early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=early_stopping_patience, restore_best_weights=True)
  callbacks_list.append(early_stopping_cb)

  # Model Chekckpoint callback
  CKPT_path = ckpt_path_name
  checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)
  callbacks_list.append(checkpointing_cb)

  return callbacks_list