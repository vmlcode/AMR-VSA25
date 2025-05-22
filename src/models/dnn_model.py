import tensorflow as tf

def build_dnn_model(name, hidden_layers, optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"], regularization=None):
    """
    Builds a sequential Keras model.
    Args:
        name: Name of the model.
        params: List of dicts with 'units' for each Dense layer.
        optimizer: Optimizer name.
        loss: Loss function.
        metrics: List of metrics.
        regularization: 'L1' or 'L2' or None.
    """
    model = tf.keras.Sequential(name=name)

    if regularization == "L1":
        regularizer = tf.keras.regularizers.l1(0.01)
    elif regularization == "L2":
        regularizer = tf.keras.regularizers.l2(0.01)
    else:
        regularizer = None

    for i in range(len(hidden_layers)):
        model.add(tf.keras.layers.Dense(
            units=hidden_layers[i]["units"],
            activation="relu",
            kernel_regularizer=regularizer,
            name=f"DENSE_{i+1}_{hidden_layers[i]['units']}_Relu"
        ))
    
    model.add(tf.keras.layers.Dense(1, activation="sigmoid", name=f"DENSE_{len(hidden_layers)+1}_1_Sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.BinaryCrossentropy(), metrics=metrics)
    
    return model
