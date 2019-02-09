from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import pandas as pd


train_dir = "./data/balanced_trainset"
validation_dir = "./data/Validation_set"
test_dir = "./data/Validation_set/"
model_save_name = "lastt_50epoch_relu_vgg16_imnet_Dense64"
acc_figure_title = "_VGG16 Accuracy - Imnet Transfer Learning"
loss_figure_title = "_VGG16 Loss - Imnet Transfer Learning"
do_save = False
num_epoch = 50

#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)


#%% Construct the whole model
# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(7, activation='softmax'))

# Make model parallel.
#model = multi_gpu_model(model, gpus=7)

# Show a summary of the model. Check the number of trainable parameters
model.summary()

#%% Setup Data Generator

# train_dir = "/home/burak/oguz_data/data/DATASET/balanced_trainset"  # Server trainset path.

# validation_dir = "/home/burak/oguz_data/data/DATASET/Validation_set"  # Server validationset path.

# train_batchsize = int(input('Enter train batch size: '))

train_batchsize = 18
val_batchsize = 8

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)
'''                                     rotation_range=20,
                                        zoom_range=0.2,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')
'''

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=train_batchsize,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)
print('\nData Generators are ready---------')

#%% Train the Model.

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['acc'])

# epok = int(input('Enter the number of epochs: '))

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=num_epoch,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    verbose=1)

print('\nTraining completed---------')
# Save the model
if do_save:
    model.save("/home/burak/oguz_data/models/" + model_save_name + ".h5")

# print('Model Saved as small_last4.h5')

#%% Evaluate model.

print('\nEvaluation Started------------------')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode='categorical',
    shuffle=False,
    seed=42
)
print('Test generator created. Prediction is starting..')

test_generator.reset()
pred=model.predict_generator(test_generator,verbose=1)

print('Saving prediction results..')
pd.DataFrame(pred, columns=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']).to_csv('vgg16_TL_50epoch_prediction.csv')

print('Saved.')

# Save Plots
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('VGG16 Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig("/home/burak/oguz_data/out_graphs/" + acc_figure_title +  ".png", bbox_inches='tight')


plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('VGG16 Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig("/home/burak/oguz_data/out_graphs/" + loss_figure_title +  ".png", bbox_inches='tight')

#plt.show()
print('Plots are saved.')

print('Process finished.')
