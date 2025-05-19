import tensorflow as tf

def build_cnn_model(name, hidden_layers, optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.F1Score()], regularization=None):
    """
    Builds a sequential Keras model.
    Args:
        name: Name of the model.
        hidden_layers: List of dicts with 'filters' and 'kernel_size' for each Conv2D layer.
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
        if hidden_layers[i]["layerType"] == "Conv2D":
            model.add(tf.keras.layers.Conv2D(
                filters=hidden_layers[i]["filters"],
                kernel_size=hidden_layers[i]["kernel_size"],
                activation="relu",
                kernel_regularizer=regularizer,
                name=f"CONV2D_{i+1}_{hidden_layers[i]['filters']}_Relu",
                input_shape=hidden_layers[i]["input_shape"],
            ))
        elif hidden_layers[i]["layerType"] == "MaxPooling2D":
            model.add(tf.keras.layers.MaxPooling2D(
                pool_size=hidden_layers[i]["pool_size"],
                name=f"MAXPOOLING2D_{i+1}_{hidden_layers[i]['pool_size'][0]}x{hidden_layers[i]['pool_size'][1]}",
            ))
        elif hidden_layers[i]["layerType"] == "Flatten":
            model.add(tf.keras.layers.Flatten(
                name=f"FLATTEN_{i+1}",
            ))
        elif hidden_layers[i]["layerType"] == "Dense":
            model.add(tf.keras.layers.Dense(
                units=hidden_layers[i]["units"],
                activation="relu",
                kernel_regularizer=regularizer,
                name=f"DENSE_{i+1}_{hidden_layers[i]['units']}_Relu"
            ))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid", name=f"DENSE_{len(hidden_layers)+1}_1_Sigmoid"))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

print(build_cnn_model("cnn_3456_45", [{"layerType": "Conv2D", "filters": 32, "kernel_size": (3, 3), "input_shape": (64, 64, 1)}, {"layerType": "MaxPooling2D", "pool_size": (2, 2)}, {"layerType": "Flatten"}, {"layerType": "Dense", "units": 128}]).summary())