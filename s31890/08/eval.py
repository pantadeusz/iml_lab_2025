import data
import keras

train_ds, test_ds, info = data.get_mnist()
data.print_info(info)

BATCH_SIZE = 128
test_ds = data.prepare(test_ds, info, batch_size=BATCH_SIZE, augment=True)

loaded_model = keras.saving.load_model("best_tuner_model_no_augment.keras")

results = loaded_model.evaluate(test_ds)

print("test loss, test acc: ", results)
