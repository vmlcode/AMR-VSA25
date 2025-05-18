import tensorflow as tf

def build_model(name, params):
    model = tf.keras.Sequential(name=name)


    for i in range(len(params)):
        model.add(tf.keras.layers.Dense(
            units=params[i]["units"],
            activation="relu",
        ))
    
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    
    print("Model layers:", model.layers)
    
    return model


test_params = [{"units": 6}, {"units": 5}, {"units": 4}]
print(build_model())
