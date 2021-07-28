# import flwr as fl

# # Start Flower server for three rounds of federated learning
# fl.server.start_server(config={"num_rounds": 3})
import flwr as fl
from pandas.core.arrays.integer import Int64Dtype
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

x_pred=pd.DataFrame(columns=["name","km_driven","fuel","seller_type","transmission","owner","years"])

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    # (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    df=pd.read_csv("car rental/CAR DETAILS FROM CAR DEKHO.csv")
    df.dropna(inplace=True)
    df.replace("",np.nan,inplace=True)

    df.dropna(axis=0,how="any",inplace=True)
    encode=LabelEncoder()
    for j in ["fuel","seller_type","transmission","owner","name"]:
        df[j]=encode.fit_transform(df[j])
    y=df.iloc[:,1]
    x=df.drop("selling_price",inplace=False,axis=1)
    x_train,xtest,y_train,ytest =train_test_split(x.values[:500],y.values[:500],test_size=0.25,random_state=100)
    x_train = x_train.reshape(-1, 7)
    # x_pred=x_train
    # x_pred.columns=x_train.columns



    # Use the last 5k training examples as a validation set
    x_val, y_val = x_train[100:], y_train[100:]

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights, ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        print("Size ",(x_val[1].shape))
        print("Xval looks",x_val[1])
        return loss, {"mae": accuracy}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}



# Start Flower server for three rounds of federated learning
# if __name__ == "__main__": 
# model = tf.keras.applications.EfficientNetB0(
#         input_shape=(32, 32, 3), weights=None, classes=10
#     )
model=tf.keras.models.Sequential([tf.keras.layers.Dense(units=64,input_shape=(7,),weights=None,activation="linear")])
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

strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_eval=0.2,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()))
def start(strategy):
    fl.server.start_server("localhost:3000", config={"num_rounds": 3}, strategy=strategy)
   
start(strategy)
f=np.array([133,149000,1,1,1,2,10])
# # f=pd.Series(f)
# x_pred.loc[len(x_pred)+1]=np.array([133,149000,1,1,1,2,10])
# print(x_pred.iloc[-1].shape)
f=f.reshape(-1,7)
print(f.shape)
z=model.predict(f)
print(z)