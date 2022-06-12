import tensorflow as tf
from tensorflow.keras import backend, optimizers
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from keras.applications.inception_v3 import InceptionV3
import os

# Define variables
img_dir = '../data'
train_dir = os.path.join(img_dir, 'Training')
test_dir = os.path.join(img_dir, 'Test')

BATCH_SIZE = 32
IMG_SIZE = (299, 299)

# Load datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  shuffle=True,
  batch_size=BATCH_SIZE,
  image_size=IMG_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  shuffle=True,
  batch_size=BATCH_SIZE,
  image_size=IMG_SIZE
)

# Pre-process the data
train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))
validation_dataset = validation_dataset.map(lambda x, y: (x / 255.0, y))

train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


# Start training
backend.clear_session()

# Create base model
conv_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Add the fully-connected layers
InceptionV3_model_input = conv_base.output
pooling = GlobalAveragePooling2D()(InceptionV3_model_input)
dense = Dense(512, activation='relu')(pooling)
output = Dense(131, activation='softmax')(dense)

model_InceptionV3 = Model(inputs=conv_base.input, outputs=output)
# model_InceptionV3.summary()

# Compile and train the model
model_InceptionV3.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9),
    metrics=['accuracy']
)

history = model_InceptionV3.fit(
    train_dataset,
    epochs=5,
    validation_data=validation_dataset,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)]
)

# Save the model
model_InceptionV3.save('model_name.h5')