import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Tuple

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

	val_ds_pipeline = val_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
	val_ds_pipeline = val_ds_pipeline.batch(batch_size)
	val_ds_pipeline = val_ds_pipeline.prefetch(AUTOTUNE)

	test_ds_pipeline = test_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
	test_ds_pipeline = test_ds_pipeline.batch(batch_size)
	test_ds_pipeline = test_ds_pipeline.prefetch(AUTOTUNE)

	return train_ds_pipeline, val_ds_pipeline, test_ds_pipeline


def get_best_model(activation='relu',
				   optimizer='adam'):
	model = tf.keras.Sequential()
	pass

if __name__ == '__main__':
	dataset_name = 'beans'
	batch_size = 32

	train_ds, val_ds, test_ds = load_data(dataset_name)
	if not all_images_same_size(train_ds, val_ds, test_ds):
		raise Exception('Images does not have same size, resize needed.')
	
	train_ds_pipeline, val_ds_pipeline, test_ds_pipeline = get_pipelines(train_ds, val_ds, test_ds, batch_size)

	print('✨ Finished')
	# model = get_best_model()
	# model.summary()