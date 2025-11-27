import data
import keras

train_ds, test_ds, info = data.get_mnist()
data.print_info(info)

BATCH_SIZE = 128
test_ds = data.prepare(test_ds, info, batch_size=BATCH_SIZE, augment=True)

loaded_model = keras.saving.load_model("best_tuner_model_no_augment.keras")
# Retrain the same model
history = loaded_model.fit(
    train_ds, batch_size=BATCH_SIZE, epochs=30, validation_data=test_ds
)

loaded_model.save("model_retrained_augment.keras")
print("Saved the retrained model")
