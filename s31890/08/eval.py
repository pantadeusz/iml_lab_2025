import data
import keras

train_ds, test_ds, info = data.get_mnist()
data.print_info(info)

BATCH_SIZE = 128
test_ds = data.prepare(test_ds, info, batch_size=BATCH_SIZE, augment=True)

loaded_model = keras.saving.load_model("best_tuner_model_no_augment.keras")
loaded_model_augment = keras.saving.load_model("model_retrained_augment.keras")

results = loaded_model.evaluate(test_ds)
results_augment = loaded_model_augment.evaluate(test_ds)

print("Standard model tested on augmented data without retraining:")
print("test loss, test acc: ", results)

print("Retrained model:")
print("test loss, test acc: ", results_augment)
