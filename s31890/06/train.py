import data
import model
import keras_tuner as kt
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Optionally set the device to CPU by disabnling GPU
# tf.config.set_visible_devices([], 'GPU')

train_ds, val_ds, test_ds, info = data.get_beans()

data.print_info(info)
train_counts, train_weights = data.compute_class_weights(train_ds)
print(f"Training dataset counts: {train_counts}")
print(f"Training dataset weights: {train_weights}")
data.print_sample_info(next(iter(train_ds.take(1))))
sample_shape = data.get_sample_shape(next(iter(train_ds.take(1))))

BATCH_SIZE = 8
train_ds = train_ds.batch(BATCH_SIZE)
val_ds = val_ds.batch(BATCH_SIZE)
test_ds = test_ds.batch(BATCH_SIZE)


def build_model(hp):
    return model.create_model_with_params(
        hp=hp,
        input_shape=sample_shape,
        num_classes=3,
        num_conv_blocks=hp.Int('num_conv_blocks', 2, 3),
        conv_filters=[32, 64, 128],
        kernel_size=(3, 3),
        activation=hp.Choice('activation', ['relu', 'elu']),
        use_batch_norm=hp.Boolean('use_batch_norm'),
        num_dense_layers=hp.Int('num_dense_layers', 2, 4),
        units_per_dense=hp.Int('units_per_dense', 64, 512, step=64),
        dropout_rate=hp.Float('dropout_rate', 0.2, 0.7, step=0.1),
        learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log'),
        optimizer=hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
    )

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=30,
    factor=3,
    directory='tuner_cache',
    project_name='beans_tuning',
    hyperband_iterations=2
)

# Do not search, because the models are already in the cache. Reload instead. If you wish to train change this line to search
tuner.reload()

best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

best_model.save("best_tuner_model.keras")

print("Best hyperparameters:")
print(best_hps.values)

data.save_tuner_summary(tuner, num_trials=len(tuner.oracle.trials))
