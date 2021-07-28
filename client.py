

import argparse
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd;
from numpy import random

import tensorflow as tf

import flwr as fl

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        # results = {
        #     "loss": history.history["loss"][0],
        #     "accuracy": history.history["accuracy"][0],
        #     "val_loss": history.history["val_loss"][0],
        #     "val_accuracy": history.history["val_accuracy"][0],
        # }
        results ={
            "loss":history.history["loss"][0]
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"mae": accuracy}
    def find(self):
        f=np.array([133,149000,1,1,1,2,10])
        f=f.reshape(-1,7)
        z=self.model.predict(f)
        print("Prediction from this client --- ",z)

def main() -> None:
    # Parse command line argument `partition`
    # parser = argparse.ArgumentParser(description="Flower")
    # parser.add_argument("--partition", type=int, choices=range(0, 10), required=False)
    # args = parser.parse_args(1)

    # Load and compile Keras model
    # model = tf.keras.applications.EfficientNetB0(
    #     input_shape=(32, 32, 3), weights=None, classes=10
    # )
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model=tf.keras.models.Sequential([tf.keras.layers.Dense(units=64,input_shape=(7,),activation="linear")])
    model.add(tf.keras.layers.Dense(units=32,activation="relu"))
    model.add(tf.keras.layers.Dense(units=32,activation="linear"))
    model.add(tf.keras.layers.Dense(units=16,activation="relu"))
    model.add(tf.keras.layers.Dense(units=16,activation="linear"))
    model.add(tf.keras.layers.Dense(activation="linear",units=8))
    model.add(tf.keras.layers.Dense(activation="linear",units=1))
# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.compile(loss="mae",optimizer=
                      # tf.keras.optimizers.RMSprop()
              tf.keras.optimizers.Adam(0.01)
              ,metrics=['mae'])



    df=pd.read_csv("car rental/CAR DETAILS FROM CAR DEKHO.csv")
    df.dropna(inplace=True)
    
    df.replace("",np.nan,inplace=True)

    df.dropna(axis=0,how="any",inplace=True)
    encode=LabelEncoder()
    for j in ["fuel","seller_type","transmission","owner","name"]:
        df[j]=encode.fit_transform(df[j])
    y=df.iloc[:,1]
    x=df.drop("selling_price",inplace=False,axis=1)

    n=np.random.randint(20,len(x))
    print("Number of samples for training! ",n)
    x_train,x_test,y_train,y_test=train_test_split(x.values[550:],y.values[550:],test_size=0.25,random_state=10)

    # (x_train, y_train), (x_test, y_test) = load_partition(
    #     #args.partition
    #     8)

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("localhost:3000", client=client)
    client.find()


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (
        x_train[idx * 5000 : (idx + 1) * 5000],
        y_train[idx * 5000 : (idx + 1) * 5000],
    ), (
        x_test[idx * 1000 : (idx + 1) * 1000],
        y_test[idx * 1000 : (idx + 1) * 1000],
    )


if __name__ == "__main__":
    main()
    