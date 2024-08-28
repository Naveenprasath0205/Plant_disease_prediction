import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
import scipy

# Define constants
IMAGE_SIZE = (299, 299)
BATCH_SIZE = 32
NUM_CLASSES = 10
NUM_EPOCHS = 3

train_data_directory = "E:/Studies/datasets/tomato/train"
test_data_directory = "E:/Studies/datasets/tomato/val"

# Data Augmentation and Image Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(
    train_data_directory,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = train_datagen.flow_from_directory(
    test_data_directory,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Define the model
def get_model():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    for layer in base_model.layers:
        layer.trainable = False

    base_model_output = base_model.output

    x = Flatten()(base_model_output)
    x = Dense(512, activation='relu')(x)
    x = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model

# Get the model
model = get_model()

# Compile the model
model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model using fit
history = model.fit(
    train_generator,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator
)

# Evaluate the model on the validation set
evaluation_results = model.evaluate(validation_generator)

# Display evaluation results
print("Evaluation Results:")
print(f"Loss: {evaluation_results[0]}")
print(f"Accuracy: {evaluation_results[1]}")

# Save the model
model.save('plant_disease_classifier_model.h5')
