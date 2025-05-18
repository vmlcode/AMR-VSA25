
def evaulate_model(model, x_test, y_test):
    """ return statistics of the model on the test set, loss of the train and test dataset and also accuracy """
    """
    Args:
        model: Keras model.
        x_test: Test data.
        y_test: Test labels.
    """
    return model.evaluate(x_test, y_test)
