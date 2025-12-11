import data
import model
import keras_tuner as kt
import tensorflow as tf
import matplotlib.pyplot as plt
import keras

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# Optionally set the device to CPU by disabnling GPU
# tf.config.set_visible_devices([], 'GPU')

train_ds, test_ds, info = data.get_imdb()
data.print_info(info)

BATCH_SIZE = 128
train_ds = data.prepare(train_ds, info, batch_size=BATCH_SIZE, shuffle=True)
test_ds = data.prepare(test_ds, info, batch_size=BATCH_SIZE)


VOCAB_SIZE = 1000
encoder = keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_ds.map(lambda text, label: text))

def build_model(hp):
    num_dense_layers=hp.Int("num_dense_layers", 1, 3)
    dense_shapes=[hp.Int(f"units_dense_{i}", 32, 128, step=32) for i in range(num_dense_layers)]

    num_bidirectional_layers=hp.Int("num_bidirectional_layers", 1, 3)
    bidirectional_shapes=[hp.Int(f"units_bidirectional_{i}", 32, 128, step=32) for i in range(num_bidirectional_layers)]

    return model.create_model_with_params(
        encoder=encoder,
        activation=hp.Choice("activation", ["relu", "elu"]),
        use_batch_norm=hp.Boolean("use_batch_norm"),
        dense_shapes=dense_shapes,
        bidirectional_shapes=bidirectional_shapes,
        dropout_rate=hp.Float("dropout_rate", 0.2, 0.7, step=0.1),
        learning_rate=hp.Float("learning_rate", 1e-4, 1e-2, sampling="log"),
        optimizer=hp.Choice("optimizer", ["adam", "sgd", "rmsprop"]),
    )

# hp = kt.HyperParameters()
#
# hp.values["num_dense_layers"] = 2
# hp.values["units_dense_1"] = 64
# hp.values["units_dense_2"] = 32
# hp.values["num_bidirectional_layers"] = 2
# hp.values["units_bidirectional_1"] = 64
# hp.values["units_bidirectional_2"] = 32
# hp.values["activation"] = "relu"
# hp.values["use_batch_norm"] = True
# hp.values["dropout_rate"] = 0.5
# hp.values["learning_rate"] = 0.001
# hp.values["optimizer"] = "adam"
#
# test_model = build_model(hp)
# test_model.summary()
# history = test_model.fit(train_ds, epochs=10, validation_data=test_ds, validation_steps=30)
# test_loss, test_acc = test_model.evaluate(test_ds)
# print('Test Loss:', test_loss)
# print('Test Accuracy:', test_acc)
#
# def plot_graphs(history, metric):
#   plt.plot(history.history[metric])
#   plt.plot(history.history['val_'+metric], '')
#   plt.xlabel("Epochs")
#   plt.ylabel(metric)
#   plt.legend([metric, 'val_'+metric])
#
# plt.figure(figsize=(16, 8))
# plt.subplot(1, 2, 1)
# plot_graphs(history, 'accuracy')
# plt.ylim(None, 1)
# plt.subplot(1, 2, 2)
# plot_graphs(history, 'loss')
# plt.ylim(0, None)
# plt.savefig("test_graph.png")

tuner = kt.Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=30,
    factor=3,
    directory="tuner_cache",
    project_name="imdb_tuning",
    hyperband_iterations=1,
)

tuner.search(train_ds, validation_data=test_ds, epochs=30)

best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

best_model.save("best_tuner_model.keras")

print("Best hyperparameters:")
print(best_hps.values)

data.save_tuner_summary(tuner, num_trials=len(tuner.oracle.trials))
