from glob import glob

import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime, os
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from IPython.display import clear_output
import tensorflow_addons as tfa

IMG_SIZE = 128
N_CHANNELS = 3
N_CLASSES = 151 # 150 classes + "not labelled"

def create_source_dataset():


TRAINSET_SIZE = len(glob(dataset_path + training_data + "*.jpg"))
print(f"The Training Dataset contains {TRAINSET_SIZE} images.")

VALSET_SIZE = len(glob(dataset_path + val_data + "*.jpg"))
print(f"The Validation Dataset contains {VALSET_SIZE} images.")

def parse_image(img_path: str) -> dict:
  """Load an image and its annotation (mask) and returning
  a dictionary.

  Parameters
  ----------
  img_path : str
    Image (not the mask) location.

  Returns
  -------
  dict
    Dictionary mapping an image and its annotation.
  """
  image = tf.io.read_file(img_path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.uint8)

  # For one Image path:
  # .../trainset/images/training/ADE_train_00000001.jpg
  # Its corresponding annotation path is:
  # .../trainset/annotations/training/ADE_train_00000001.png
  mask_path = tf.strings.regex_replace(img_path, "images", "annotations")
  mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
  mask = tf.io.read_file(mask_path)
  # The masks contain a class index for each pixels
  mask = tf.image.decode_png(mask, channels=1)
  # In scene parsing, "not labeled" = 255
  # But it will mess up with our N_CLASS = 150
  # Since 255 means the 255th class
  # Which doesn't exist
  mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)
  # Note that we have to convert the new value (0)
  # With the same dtype than the tensor itself

  return {'image': image, 'segmentation_mask': mask}

if __name__ == "__main__":
    
  dataset_path = "data/Training/images/"
  training_data = "training/"
  val_data = "validation/"