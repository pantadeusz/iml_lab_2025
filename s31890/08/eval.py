import data
import keras

train_ds, test_ds, info = data.get_mnist()
data.print_info(info)

BATCH_SIZE = 128
test_ds_augment = data.prepare(test_ds, info, batch_size=BATCH_SIZE, augment=True)
test_ds = data.prepare(test_ds, info, batch_size=BATCH_SIZE, augment=False)

loaded_model = keras.saving.load_model("best_tuner_model_no_augment.keras")
loaded_model_augment = keras.saving.load_model("model_retrained_augment.keras")

base_results = loaded_model.evaluate(test_ds)
base_results_augment = loaded_model.evaluate(test_ds_augment)

retrained_results = loaded_model_augment.evaluate(test_ds)
retrained_results_augment = loaded_model_augment.evaluate(test_ds_augment)

print("Base model:")
print("Standard MNIST: test loss, test acc: ", base_results)
print("Augmented: test loss, test acc: ", base_results_augment)

print("Retrained model:")
print("Standard MNIST: test loss, test acc: ", retrained_results)
print("Augmented: test loss, test acc: ", retrained_results_augment)
