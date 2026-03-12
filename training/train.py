import numpy as np
import tensorflow as tf
import keras
from utils import DataGenerator, get_matchdataset, get_dataset_c1
from pathlib import Path

Path("models").mkdir(exist_ok=True)

tf.config.experimental.enable_tensor_float_32_execution(False)

#load data
matchdat = np.load("matchinput.npy")
metadat = np.load("metainput.npy")
mdat = get_matchdataset(matchdat) 
trainingGenerator = DataGenerator(metadat)
c1set = get_dataset_c1(trainingGenerator)

#construct the model
profileInputs = {"rho": keras.Input(shape=(401,), name="rho")}
phiInputs = {"phi": keras.Input(shape= (150,), name = "phi")}

all_inputs = profileInputs | phiInputs

x = keras.layers.Concatenate()(list(all_inputs.values()))

x = keras.layers.Dense(512, activation="softplus")(x)
x = keras.layers.Dense(256, activation="softplus")(x)
x = keras.layers.Dense(128, activation="softplus")(x)
x = keras.layers.Dense(64, activation="softplus")(x)
x = keras.layers.Dense(32, activation="softplus")(x)

outputs = {"c1": keras.layers.Dense(1, name="c1")(x)}

model = keras.Model(inputs=all_inputs, outputs=outputs)

optimizer = keras.optimizers.Adam()
loss = keras.losses.MeanSquaredError()
metrics = [keras.metrics.MeanAbsoluteError()]
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics,
)
model.summary()

# define training step
@tf.function
def train_step(x_c1, y_c1, x_c2x=None, y_c2x=None, alpha_c1=1, alpha_c2x=0.01):
    with tf.GradientTape() as tape:
        c1_model = model(x_c1, training=True)["c1"]
        loss_c1 = loss(y_c1["c1"], c1_model)
        loss_c2x = 0
        if alpha_c2x > 0:
            with tf.GradientTape(watch_accessed_variables=False) as tape2:  
                tape2.watch(x_c2x["phi"])
                c1_model_match = model(x_c2x)["c1"] 
            dc1_dphi = tape2.gradient(c1_model_match, x_c2x["phi"])*401.0 # current normalization of training data, check if you use own data
            loss_c2x = loss(y_c2x["cphi"], dc1_dphi)
        loss_total = alpha_c1 * loss_c1 + alpha_c2x * loss_c2x
    grads = tape.gradient(loss_total, model.trainable_weights)
    optimizer.apply(grads, model.trainable_weights)
    for metric in metrics:
        metric.update_state(y_c1["c1"], c1_model)
    return loss_c1, loss_c2x
#train the model
for epoch in range(300):
    print(f"Epoch: {epoch}")
    print(f"\tlearning rate: {optimizer.learning_rate.numpy():.4g}")


    for step, ((x_c1, y_c1), (x_c2x, y_c2x)) in enumerate(zip(c1set, mdat)):
        loss_c1, loss_c2x = train_step(x_c1, y_c1, x_c2x, y_c2x, alpha_c1=1, alpha_c2x=0.01)


    print(f"\tsteps: {step}")
    print(f"\tloss_c1: {loss_c1:.4g}")
    print(f"\tloss_c2x: {loss_c2x:.4g}")

    for metric in metrics:
        print(f"\t{metric.name} (c1): {metric.result():.4g}")
        metric.reset_state()

    model.save("models/meta_match_test.keras")
    optimizer.learning_rate *= 0.98

weights = model.get_weights()
np.savez("new_model_weights.npz", *weights)#save weights of the model 
