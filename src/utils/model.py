import tensorflow as tf

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NO_OF_CLASSES):
    LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28,28], name="input_layer"),
          tf.keras.layers.Dense(units=300,activation="relu", name="hidden_layer1"),
          tf.keras.layers.Dense(units=100,activation="relu", name="hiddel_layer2"),
          tf.keras.layers.Dense(units=NO_OF_CLASSES,activation="softmax", name="output_layer")
          ]
    
    # Creating model 
    model_clf = tf.keras.models.Sequential(LAYERS)

    model_clf.summary() 

    model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

    return model_clf ## returns an untrained model