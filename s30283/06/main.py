import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from typing import Tuple
import logging

DatasetType = tf.data.Dataset

def load_data(dataset_name: str) -> Tuple[DatasetType, DatasetType, DatasetType]:
	(train_ds, val_ds, test_ds) = tfds.load(dataset_name, 
										    split=['train', 'validation', 'test'],
										    shuffle_files=True,
											as_supervised=True)
	return train_ds, val_ds, test_ds

def all_images_same_size(train_ds: DatasetType, 
						 val_ds: DatasetType, 
						 test_ds: DatasetType) -> bool:
	return all([all(item[0].shape == (500, 500, 3) for item in ds) for ds in [train_ds, val_ds, test_ds]])

def preprocess_image(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
	image = tf.cast(image, tf.float32)
	return image / 255.0, label

def get_pipelines(train_ds: DatasetType,
				  val_ds: DatasetType, 
				  test_ds: DatasetType, 
				  batch_size=32) -> Tuple[DatasetType, DatasetType, DatasetType]:
	"""
	- every ds holds a list of file paths on disk like /.../img.jpg, so that it does no use RAM
	- AUTOTUNE enables working in parallel
	- shuffling - we can't load all 1000+ images into RAM to shuffle them, so it creates a buffer of 1024 items. It pulls 1024 pre-processed images from previous stage and puts in a "bin". When next stage requests intem .shuffle() randomly picks one from the bin, sends it and pulls a new item from previous stage to replace it
	- .batch() packs items together into batches
	- .prefetch() - while gpu is busy while training on batch 1 then cpu is already working to prepare batch 2
	"""

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds_pipeline = train_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    train_ds_pipeline = train_ds_pipeline.shuffle(1024)
    train_ds_pipeline = train_ds_pipeline.batch(batch_size)
    train_ds_pipeline = train_ds_pipeline.prefetch(AUTOTUNE)
    print(list(zip(train_ds_pipeline)))
    exit(0)
	val_ds_pipeline = val_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
	val_ds_pipeline = val_ds_pipeline.batch(batch_size)
	val_ds_pipeline = val_ds_pipeline.prefetch(AUTOTUNE)

	test_ds_pipeline = test_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
	test_ds_pipeline = test_ds_pipeline.batch(batch_size)
	test_ds_pipeline = test_ds_pipeline.prefetch(AUTOTUNE)

	return train_ds_pipeline, val_ds_pipeline, test_ds_pipeline

def build_model(hp: kt.HyperParameters, 
                input_shape: Tuple[int, int, int],
				activation: str,
				optimizer: str) -> tf.keras.Sequential:
    """
    This is the model-building function that KerasTuner will use.
    'hp' is an object that allows you to define tunable hyperparameters.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(
        units=hp.Int('units_layer1', min_value=8, max_value=128, step=8),
        activation=activation,
    ))
    model.add(tf.keras.layers.Dense(
        units=hp.Int('units_layer2', min_value=4, max_value=64, step=4),
        activation=activation
    ))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    
    learning_rate = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')

    final_optimizer = None
    if optimizer == 'adam':
        final_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        final_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        final_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise Exception('Specified unavailable optimizer. Choose from: adam, sgd, rmsprop')

    model.compile(
        optimizer=final_optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def tune_and_train_nn_model(train_ds: DatasetType, 
                            val_ds: DatasetType,
							activation: str,
							optimizer: str,
							input_shape: Tuple[int, int, int]) -> tf.keras.Sequential:
    model_builder = lambda hp: build_model(hp, 
	                                       input_shape=input_shape,
										   activation=activation,
										   optimizer=optimizer)

    tuner = kt.RandomSearch(
        model_builder,
        objective='val_accuracy',  # What to maximize
        max_trials=50,             # How many different models to try
        executions_per_trial=1,    # How many times to train each model
        directory='tuning_dir',
        project_name='beans_tuning'
    )
    
    stop_early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    logging.info('🚀 Starting hyperparameter search...')
    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[stop_early],
        verbose=1
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    logging.info(f"""
    ✅ Tuning complete. Best hyperparameters found:
    - Layer 1 Units: {best_hps.get('units_layer1')}
    - Layer 2 Units: {best_hps.get('units_layer2')}
    - Learning Rate: {best_hps.get('lr'):.5f}
    """)

    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model

if __name__ == '__main__':
	dataset_name = 'beans'
	batch_size = 16

	train_ds, val_ds, test_ds = load_data(dataset_name)
	if not all_images_same_size(train_ds, val_ds, test_ds):
		raise Exception('Images does not have same size, resize needed.')
	
	train_ds_pipeline, val_ds_pipeline, test_ds_pipeline = get_pipelines(train_ds, val_ds, test_ds, batch_size)

	best_model = tune_and_train_nn_model(train_ds_pipeline, 
	                                     val_ds_pipeline,
										 activation='relu',
										 optimizer='adam',
										 input_shape=(500, 500, 3))
	best_model.summary()

	print('✨ Finished')