import keras
import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

np.random.seed(15)  # for reproducibility


def prep_train_test(X_train, X_test):
    """
    Prep samples ands labels for Keras input by noramalzing and converting
    labels to a categorical representation.
    """
    print(
        "Train on {} samples, validate on {}".format(X_train.shape[0], X_test.shape[0])
    )

    # normalize to dBfS
    X_train = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_train])
    X_test = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_test])

    return X_train, X_test


def keras_img_prep(X_train, X_test, img_rows, img_cols):
    """
    Reshape feature matrices for Keras' expexcted input dimensions.
    Tensorflow order "channels_last" (# images, # rows, # cols, # channels).
    """
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    return X_train, X_test, input_shape


def cnn(
    X_train,
    y_train,
    X_test,
    y_test,
    input_shape,
    config,
):
    """
    The Convolutional Neural Net architecture for classifying the audio clips
    as normal (0) or depressed (1).
    """
    model = Sequential(
        [
            keras.Input(shape=input_shape),
            Conv2D(32, 3, padding="same", activation=config.activation),
            MaxPooling2D(2),
            Dropout(0.5),
            Conv2D(32, 3, padding="same", activation=config.activation),
            MaxPooling2D(2),
            Dropout(0.5),
            Conv2D(32, 3, padding="same", activation=config.activation),
            MaxPooling2D(2),
            Dropout(0.5),
            Conv2D(32, 3, padding="same", activation=config.activation),
            MaxPooling2D(2),
            Dropout(0.5),
            Flatten(),
            Dense(128, activation=config.activation),
            Dense(config.num_classes, activation="softmax"),
        ]
    )
    # model = Sequential()
    # model.add( Conv2D( 32, (3, 3), padding="valid", strides=1, input_shape=input_shape, activation="relu", ) )
    # model.add(MaxPooling2D(pool_size=(4, 3), strides=(1, 3)))
    # model.add( Conv2D( 32, (1, 3), padding="valid", strides=1, input_shape=input_shape, activation="relu", ) )
    # model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    # model.add(Flatten())
    # model.add(Dense(512, activation="relu"))
    # model.add(Dense(512, activation="relu"))
    # model.add(Dropout(0.5))
    # model.add(Dense(config.num_classes))
    # model.add(Activation("softmax"))
    model.compile(loss=config.loss, optimizer=config.optimizer, metrics=[config.metric])

    history = model.fit(
        X_train,
        y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        verbose=1,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=[WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")],
    )

    # Evaluate accuracy on test and train sets
    score_train = model.evaluate(X_train, y_train, verbose=0)
    print("Train accuracy:", score_train[1])
    score_test = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy:", score_test[1])
    return model, history


def model_performance(model, X_train, X_test, y_test):
    """
    Evaluation metrics for network performance.
    """
    y_test_pred = model.predict_classes(X_test)
    y_train_pred = model.predict_classes(X_train)

    y_test_pred_proba = model.predict_proba(X_test)
    y_train_pred_proba = model.predict_proba(X_train)

    # Converting y_test back to 1-D array for confusion matrix computation
    y_test_1d = y_test[:, 1]

    # Computing confusion matrix for test dataset
    conf_matrix = standard_confusion_matrix(y_test_1d, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_train_pred, y_test_pred, y_train_pred_proba, y_test_pred_proba, conf_matrix


def standard_confusion_matrix(y_test, y_test_pred):
    """
    Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    return np.array([[tp, fp], [fn, tn]])
