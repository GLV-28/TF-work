import numpy as np
import os
import PIL
import PIL.Image
import pathlib
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets, layers, models
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import tensorflow_docs.plots
import random
import IPython
import kerastuner as kt
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
#device_count = {'GPU': 1}
)
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)

config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def myprint(s):
    with open('modelsummary.txt','w+') as f:
        print(s, file=f)



#datasets made through tf.directory function
train=os.listdir("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/New-Dataset/Training")
test=os.listdir("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/New-Dataset/Test")

num_train=len(train)
num_test=len(test)
x_train = np.empty((num_train, 3, 32, 32), dtype='uint8')
y_train = np.empty((num_train,), dtype='uint8')
train_f=("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/New-Dataset/Training")
test_f=("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/New-Dataset/Test")
Sizer1=(8,8)
Sizer2=(16,16)
SizerF=(32,32)
#8x8 Dataset
train_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    train_f, labels='inferred', label_mode='int', class_names=None,
    color_mode='rgb', batch_size=32, image_size=(Sizer1), shuffle=True, seed=339,
    validation_split=0.2, subset="training", interpolation='bilinear', follow_links=False
)

val_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    train_f, labels='inferred', label_mode='int', class_names=None,
    color_mode='rgb', batch_size=32, image_size=(Sizer1), shuffle=True, seed=339,
    validation_split=0.2, subset="validation", interpolation='bilinear', follow_links=False
)


test_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    test_f, labels='inferred', label_mode='int', class_names=None,
    color_mode='rgb', batch_size=32, image_size=(Sizer1), shuffle=True, seed=None,
    validation_split=None, subset=None, interpolation='bilinear', follow_links=False
)
class_names = train_dataset.class_names
print(class_names)
num_classes=int(len(class_names))
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dsS1 = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dsS1 = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dsS1 = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)


#16x16 Dataset
train_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    train_f, labels='inferred', label_mode='int', class_names=None,
    color_mode='rgb', batch_size=32, image_size=(Sizer2), shuffle=True, seed=339,
    validation_split=0.2, subset="training", interpolation='bilinear', follow_links=False
)

val_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    train_f, labels='inferred', label_mode='int', class_names=None,
    color_mode='rgb', batch_size=32, image_size=(Sizer2), shuffle=True, seed=339,
    validation_split=0.2, subset="validation", interpolation='bilinear', follow_links=False
)


test_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    test_f, labels='inferred', label_mode='int', class_names=None,
    color_mode='rgb', batch_size=32, image_size=(Sizer2), shuffle=True, seed=None,
    validation_split=None, subset=None, interpolation='bilinear', follow_links=False
)
class_names = train_dataset.class_names
print(class_names)
num_classes=int(len(class_names))
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dsS2 = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dsS2 = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dsS2 = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)





#32x32 Dataset
train_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    train_f, labels='inferred', label_mode='int', class_names=None,
    color_mode='rgb', batch_size=32, image_size=(SizerF), shuffle=True, seed=339,
    validation_split=0.2, subset="training", interpolation='bilinear', follow_links=False
)

val_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    train_f, labels='inferred', label_mode='int', class_names=None,
    color_mode='rgb', batch_size=32, image_size=(SizerF), shuffle=True, seed=339,
    validation_split=0.2, subset="validation", interpolation='bilinear', follow_links=False
)


test_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    test_f, labels='inferred', label_mode='int', class_names=None,
    color_mode='rgb', batch_size=32, image_size=(SizerF), shuffle=True, seed=None,
    validation_split=None, subset=None, interpolation='bilinear', follow_links=False
)
class_names = train_dataset.class_names
print(class_names)
num_classes=int(len(class_names))
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

def compile_and_fitmulti(model,name,optimizer=None,MT10label= False, DIM = (),max_epochs=10):
    if optimizer is None:
        optimizer='adam'
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    if MT10label == False and DIM == (32,32):
        history=model.fit(
            train_ds,
            steps_per_epoch=32,
            epochs=max_epochs,
            validation_data=val_ds,
            # callbacks=get_callbacks(name),
            verbose=0)
        return history

    elif MT10label == False and DIM == (16,16):
        history=model.fit(
            train_dsS2,
            steps_per_epoch=32,
            epochs=max_epochs,
            validation_data=val_dsS2,
            # callbacks=get_callbacks(name),
            verbose=0)
        return history
    elif MT10label == False and DIM == (8,8):
        history=model.fit(
            train_dsS1,
            steps_per_epoch=32,
            epochs=max_epochs,
            validation_data=val_dsS1,
            # callbacks=get_callbacks(name),
            verbose=0)
        return history
    elif MT10label == True and DIM == (16,16):
        history=model.fit(
            train_dsMT10labelx16,
            steps_per_epoch=32,
            epochs=max_epochs,
            validation_data=val_dsMT10labelx16,
            # callbacks=get_callbacks(name),
            verbose=0)
        return history
    else: #MT10label == True and DIM == (32,32):
        history=model.fit(
            train_dsMT10label,
            steps_per_epoch=32,
            epochs=max_epochs,
            validation_data=val_dsMT10label,
            # callbacks=get_callbacks(name),
            verbose=0)
        return history


#To do with Sizer1
model1=models.Sequential()
model1.add(layers.experimental.preprocessing.Rescaling(1. / 255))
model1.add(layers.Flatten(input_shape=(8,8,3)))
model1.add(layers.Dense(8, activation=tf.nn.relu)),#, input_shape=(8,8, 3)))
#model1.add(layers.Dense(256, activation='relu'))
#model1.add(layers.Dense(128, activation='relu'))

model1.add(layers.Dense(32, activation=tf.nn.relu)),
#model1.add(layers.Dense(64, activation='relu'))
#model1.add(layers.Dense(128, activation='relu'))
model1.add(layers.Dense(num_classes, activation=tf.nn.softmax)),

model1.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history1 = model1.fit(train_dsS1, epochs=10,
                    validation_data=(val_dsS1))
test_loss1, test_acc1 = model1.evaluate(test_dsS1, verbose=2)

model1.summary()
model1.summary(print_fn=myprint)
plt.plot(history1.history['accuracy'], label='accuracy')
plt.plot(history1.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.01, 1])
plt.legend(loc='lower right')
plt.savefig("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/PLOTS/MT1.jpg")
model1.summary()
model1.summary(print_fn=myprint)
#To do with Sizer1 (2.1) and Sizer2 (2.2)

#Using 8x8 images
model2_1=models.Sequential()
model2_1.add(layers.experimental.preprocessing.Rescaling(1. / 255))
model2_1.add(layers.Flatten( input_shape=(8,8, 3)))
model2_1.add(layers.Dense(32, activation='relu'))
#model1.add(layers.Dense(128, activation='relu'))
model2_1.add(layers.Dense(64, activation='relu'))
#model2.add(layers.Dense(128, activation='relu'))
#model2_1.add(layers.Flatten())
model2_1.add(layers.Dense(128, activation='relu'))
model2_1.add(layers.Dense(num_classes, activation=tf.nn.softmax))

model2_1.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history2_1 = model2_1.fit(train_dsS1, epochs=10,
                    validation_data=(val_dsS1))
test_loss2_1, test_acc2_1 = model2_1.evaluate(test_dsS1, verbose=2)

#Using 16x16 images
model2_2=models.Sequential()
model2_2.add(layers.experimental.preprocessing.Rescaling(1. / 255))
model2_2.add(layers.Flatten( input_shape=(16,16, 3)))
model2_2.add(layers.Dense(32, activation='relu'))
#model2.add(layers.Dense(128, activation='relu'))
model2_2.add(layers.Dense(64, activation='relu'))
#model2_2.add(layers.Flatten())
model2_2.add(layers.Dense(128, activation='relu'))
model2_2.add(layers.Dense(num_classes, activation=tf.nn.softmax))

model2_2.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history2_2 = model2_2.fit(train_dsS2, epochs=5,
                    validation_data=(val_dsS2))
test_loss2_2, test_acc2_2 = model2_2.evaluate(test_dsS2, verbose=2)



size_historiesSize = {}
#Fix dataset used in function under (history)
def compile_and_fit(model, name, optimizer=None, max_epochs=10):
  if optimizer is None:
    optimizer = 'adam'
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

  model.summary()

  history = model.fit(
    train_ds,
    steps_per_epoch = 32,
    epochs=max_epochs,
    validation_data=val_ds,
    #callbacks=get_callbacks(name),
    verbose=0)
  return history

def x8compile_and_fit(model, name, optimizer=None, max_epochs=10):
  if optimizer is None:
    optimizer = 'adam'
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

  model.summary()

  history = model.fit(
    train_dsS1,
    steps_per_epoch = 32,
    epochs=max_epochs,
    validation_data=val_dsS1,
    #callbacks=get_callbacks(name),
    verbose=0)
  return history


def x16compile_and_fit(model, name, optimizer=None, max_epochs=10):
  if optimizer is None:
    optimizer = 'adam'
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

  model.summary()

  history = model.fit(
    train_dsS2,
    steps_per_epoch = 32,
    epochs=max_epochs,
    validation_data=val_dsS2,
    #callbacks=get_callbacks(name),
    verbose=0)
  return history

size_historiesSize = {}

size_historiesSize['Model1'] = compile_and_fitmulti(model1, '8x8/M1',MT10label=False,DIM=(8,8))
size_historiesSize['Model2_1'] = compile_and_fitmulti(model2_1, "8x8/M2",MT10label=False,DIM=(8,8)) #model2.1
size_historiesSize['Model2_2']  = compile_and_fitmulti(model2_2, "16x16/M2", MT10label=False,DIM=(16,16))#model2.2
#put models name and principal characteristics
plotter = tfdocs.plots.HistoryPlotter(metric = 'accuracy')#, smoothing_std=10)
plotter.plot(size_historiesSize)
plt.ylim([0.05, 0.7])
plt.savefig("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/PLOTS/M1&M2_8x8vs16x16_1.jpg")
plotter.plot(size_historiesSize)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [Log Scale]")
plt.savefig("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/PLOTS/LogM1&M2_8x8vs16x16.jpg")
#16x16
model3_2=models.Sequential()
model3_2.add(layers.experimental.preprocessing.Rescaling(1. / 255))
model3_2.add(layers.Flatten(input_shape=(16,16,3)))
model3_2.add(layers.Dense(128,activation='relu'))
model3_2.add(layers.Dense(64,activation='relu'))
model3_2.add(layers.Dense(32,activation='relu'))
model3_2.add(layers.Dense(num_classes,tf.nn.softmax))

model3_2.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history3_2 = model3_2.fit(train_dsS2,epochs=5,
                     validation_data=(val_dsS2))
test_loss3_2, test_acc3_2 = model3_2.evaluate(test_dsS2,verbose=2)


model3=models.Sequential()
model3.add(layers.experimental.preprocessing.Rescaling(1. / 255))
model3.add(layers.Flatten(input_shape=(32,32,3)))
model3.add(layers.Dense(64,activation='relu'))
model3.add(layers.Dense(128,activation='relu'))
#model3.add(layers.Flatten())
model3.add(layers.Dense(256,activation='relu'))
model3.add(layers.Dense(num_classes,tf.nn.softmax))

model3.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history3 = model3.fit(train_ds,epochs=5,
                     validation_data=(val_ds))
test_loss3, test_acc3 = model3.evaluate(test_ds,verbose=2)




model2=models.Sequential()
model2.add(layers.experimental.preprocessing.Rescaling(1. / 255))
model2.add(layers.Flatten(input_shape=(32,32, 3)))
model2.add(layers.Dense(256, activation='relu'))
#model2.add(layers.Flatten())
model2.add(layers.Dense(32, activation='relu'))
model2.add(layers.Dense(num_classes))

model2.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history2 = model2.fit(train_ds, epochs=3,
                    validation_data=(val_ds))
test_loss2, test_acc2 = model2.evaluate(test_ds, verbose=2)



historiesSize = {}



historiesSize['16x16'] =    compile_and_fitmulti(model3_2, "16x16/Medium",MT10label=False,DIM=(16,16)) #model3.1
historiesSize['32x32M2']  = compile_and_fitmulti(model2, "32x32/2MLarge", MT10label=False,DIM=(32,32) ) #model3.2
historiesSize['32x32M3']  = compile_and_fitmulti(model3, "32x32/3MLarge", MT10label=False,DIM=(32,32)) #model3.2
#put models name and principal characteristics
plotter = tfdocs.plots.HistoryPlotter(metric = 'accuracy')
plotter.plot(historiesSize)
plt.ylim([0.6, 1])
plt.savefig("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/PLOTS/M2&M3_16x16vs32x32_1.jpg")



#Tuning model of historiesSize above^
def model_builder(hp):
    model=keras.Sequential()
    model.add(layers.experimental.preprocessing.Rescaling(1. / 255))
    model.add(keras.layers.Flatten(input_shape=(32,32,3)))

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units=hp.Int('units',min_value=32,max_value=512,step=32)
    model.add(keras.layers.Dense(units=hp_units,activation='relu'))
    #hp_units1=hp1.Int('units',min_value=32,max_value=512,step=32)
    #model.add(keras.layers.Dense(units=hp_units,activation='relu'))
    #model.add(keras.layers.Dense(units=hp_units,activation='relu'))
   # model.add(keras.layers.Dense(units=hp_units,activation='relu'))
    model.add(keras.layers.Dense(10))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate=hp.Choice('learning_rate',values=[1e-2,1e-3,1e-4])
    #hp1_learning_rate=hp.Choice('learning_rate',values=[1e-2,1e-3,1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model



tuner = kt.Hyperband(model_builder,
                     objective = 'val_accuracy',
                     max_epochs = 10,
                     factor = 3,
                     directory = 'my_dir',
                     project_name = 'intro_to_kt',
                      overwrite=True)

class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)

tuner.search(train_dsMT10label, epochs = 10, validation_data = (val_dsMT10label), callbacks = [ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

model = tuner.hypermodel.build(best_hps)
historyH = model.fit(train_dsMT10label, epochs = 10, validation_data = val_ds)
model.summary()
testACC, testLOSS = model.evaluate(test_dsMT10label, verbose=1 )

plt.plot(historyH.history['accuracy'], label='accuracy')
plt.plot(historyH.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.9, 1])
plt.legend(loc='lower right')
plt.savefig("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/PLOTS/HP1.jpg")

#Through Tuning we get an incredible model with just one hidden layer 91%, two HL 95%

plotter.plot(historiesSize)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [Log Scale]")
plt.savefig("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/PLOTS/LogM2&M3_16x16vs32x32.jpg")


model4_1=models.Sequential()
#model4_1.add(layers.Dense(128,activation='relu',input_shape=(8,8,3)))
model4_1.add(layers.experimental.preprocessing.Rescaling(1. / 255)),
model4_1.add(layers.Conv2D(32,3,activation='relu',input_shape=(8,8,3))),
model4_1.add(layers.MaxPooling2D()),
model4_1.add(layers.Dense(128,activation='relu'))
model4_1.add(layers.Flatten()),
model4_1.add(layers.Dense(64,activation='relu')),
model4_1.add(layers.Dense(num_classes))

model4_1.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history4_1 = model4_1.fit(train_dsS1,epochs=5,
                     validation_data=(val_dsS1))
test_loss4_1, test_acc4_1 = model4_1.evaluate(test_dsS1,verbose=2)

model4_2=models.Sequential()
#model4_2.add(layers.Dense(128,activation='relu',input_shape=(16,16,3)))
model4_2.add(layers.experimental.preprocessing.Rescaling(1. / 255)),
model4_2.add(layers.Conv2D(32,3,activation='relu', input_shape=(16,16,3))),
model4_2.add(layers.MaxPooling2D()),
model4_2.add(layers.Dense(128,activation='relu'))
model4_2.add(layers.Flatten()),
model4_2.add(layers.Dense(64,activation='relu')),
model4_2.add(layers.Dense(num_classes))

model4_2.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history4_2 = model4_2.fit(train_dsS2,epochs=5,
                     validation_data=(val_dsS2))
test_loss4_2, test_acc4_2 = model4_2.evaluate(test_dsS2,verbose=2)



model4=models.Sequential()
#model4.add(layers.Dense(128,activation='relu',input_shape=(32,32,3)))
model4.add(layers.experimental.preprocessing.Rescaling(1. / 255)),
model4.add(layers.Conv2D(32,3,activation='relu', input_shape=(32,32,3))),
model4.add(layers.MaxPooling2D()),
model4.add(layers.Dense(128,activation='relu'))
model4.add(layers.Flatten()),
model4.add(layers.Dense(64,activation='relu')),
model4.add(layers.Dense(num_classes))

model4.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history4 = model4.fit(train_ds,epochs=5,
                     validation_data=(val_ds))
test_loss4, test_acc4 = model4.evaluate(test_ds,verbose=2)



historiesSizer={}

historiesSizer['8x8'] = compile_and_fitmulti(model4_1,"8x8/Small",MT10label=False, DIM=(8,8)) #model3.1
historiesSizer['16x16']  = compile_and_fitmulti(model4_2, "16x16/Medium",MT10label=False, DIM=(16,16),) #model3.2
historiesSizer['32x32']  = compile_and_fitmulti(model4,"32x32/Large",MT10label=False, DIM=(32,32),) #model3.2
#put models name and principal characteristics
plotter = tfdocs.plots.HistoryPlotter(metric = 'accuracy')
plotter.plot(historiesSizer)
plt.ylim([0.98, 1])
plt.savefig("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/PLOTS/M4Battle(x8vsx16vsx32).jpg")


#historiesSizer['8x8'] = x8compile_and_fit(model4_1, "8x8/Small") #model3.1
#historiesSizer['16x16']  = x16compile_and_fit(model4_2, "16x16/Medium") #model3.2
#historiesSizer['32x32']  = compile_and_fit(model4, "32x32/Large") #model3.2
#put models name and principal characteristics
#plotter = tfdocs.plots.HistoryPlotter(metric = 'accuracy')
#plotter.plot(historiesSizer)
#plt.ylim([0.85, 1])

plotter.plot(historiesSizer)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])

plt.savefig("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/PLOTS/LogM4Battle(x8vsx16vsx32).jpg")

plt.plot(history4.history['accuracy'], label='accuracy')
plt.plot(history4.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.savefig("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/PLOTS/M4.jpg")

model5=models.Sequential()
model5.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3))),
#model5.add(layers.Conv2D(32,3,activation='relu')),
model5.add(layers.MaxPooling2D()),
model5.add(layers.Conv2D(32,3,activation='relu')),
model5.add(layers.MaxPooling2D()),
#model4.add(layers.Conv2D(32,3,activation='relu')),
#model4.add(layers.MaxPooling2D()),
model5.add(layers.Dense(128,activation='relu'))
model5.add(layers.Flatten()),
model5.add(layers.Dense(64,activation='relu')),
model5.add(layers.Dense(num_classes))

model5.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history5 = model5.fit(train_ds,epochs=5,
                     validation_data=(val_ds))
test_loss5, test_acc5 = model5.evaluate(test_ds,verbose=2)

plt.plot(history5.history['accuracy'], label='accuracy')
plt.plot(history5.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.9, 1])
plt.legend(loc='lower right')
plt.savefig("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/PLOTS/M5(Epoch5).jpg")
model6=models.Sequential()
model6.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3))),
#model5.add(layers.Conv2D(32,3,activation='relu')),
model6.add(layers.MaxPooling2D()),
model6.add(layers.Conv2D(64,3,activation='relu')),
model6.add(layers.MaxPooling2D()),
model6.add(layers.Conv2D(64,3,activation='relu')),
model6.add(layers.MaxPooling2D()),
#model5.add(layers.Dense(128,activation='relu'))
model6.add(layers.Flatten()),
model6.add(layers.Dense(128,activation='relu')),
model6.add(layers.Dense(num_classes))

model6.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history6 = model6.fit(train_ds,epochs=10,
                     validation_data=(val_ds))
test_loss6, test_acc6 = model6.evaluate(test_ds,verbose=2)


plt.plot(history6.history['accuracy'], label='accuracy')
plt.plot(history6.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.95, 1])
plt.legend(loc='lower right')

plt.savefig("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/PLOTS/M6.jpg")


test_loss1, test_acc1 = model1.evaluate(test_dsS1, verbose=2)
test_loss2_1, test_acc2_1 = model2_1.evaluate(test_dsS1, verbose=2)
test_loss2_2, test_acc2_2 = model2_2.evaluate(test_dsS2, verbose=2)
test_loss2, test_acc2 = model2.evaluate(test_ds, verbose=2)
test_loss3_2, test_acc3_2 = model3_2.evaluate(test_dsS2,verbose=2)
test_loss3, test_acc3 = model3.evaluate(test_ds,verbose=2)
testACC, testLOSS = model.evaluate(test_ds, verbose=1 )
test_loss4_1, test_acc4_1 = model4_1.evaluate(test_dsS1,verbose=2)
test_loss4_2, test_acc4_2 = model4_2.evaluate(test_dsS2,verbose=2)
test_loss4, test_acc4 = model4.evaluate(test_ds,verbose=2)
test_loss5, test_acc5 = model5.evaluate(test_ds,verbose=2)
test_loss6, test_acc6 = model6.evaluate(test_ds,verbose=2)







#Refined Data generating

data_dir=train_f
data_dir = pathlib.Path(data_dir)
test_data_dir=test_f
test_data_dir=pathlib.Path(test_data_dir)
img_height= 32
img_width= 32
batch_size = 32
image_count = len(list(data_dir.glob("*/*.jpg")))
image_count_test=len(list(test_data_dir.glob("*/*.jpg")))
#print(image_count)
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)
num_classes=(int(len(class_names)))
list_ds_test = tf.data.Dataset.list_files(str(test_data_dir/'*/*'), shuffle=False)
list_ds_test = list_ds_test.shuffle(image_count_test, reshuffle_each_iteration=False)

val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)
test_ds=list_ds_test.take(int(image_count_test))
print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())
print(tf.data.experimental.cardinality(test_ds).numpy())
def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)


def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])


def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label





def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

test_ds = configure_for_performance(test_ds)

#Best model so far with better data--->better performances


model6=models.Sequential()
model6.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3))),
model6.add(layers.MaxPooling2D()),
model6.add(layers.Conv2D(64,3,activation='relu')),
model6.add(layers.MaxPooling2D()),
model6.add(layers.Conv2D(64,3,activation='relu')),
model6.add(layers.MaxPooling2D()),
model6.add(layers.Flatten()),
model6.add(layers.Dense(128,activation='relu')),
model6.add(layers.Dense(num_classes))

model6.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history6 = model6.fit(train_ds,epochs=5,
                     validation_data=(val_ds))
test_loss6, test_acc6 = model6.evaluate(test_ds,verbose=2)



model4=models.Sequential()
#model4.add(layers.Dense(128,activation='relu',input_shape=(32,32,3)))
model4.add(layers.experimental.preprocessing.Rescaling(1. / 255)),
model4.add(layers.Conv2D(32,3,activation='sigmoid', input_shape=(32,32,3))),
model4.add(layers.MaxPooling2D()),
model4.add(layers.Dense(128,activation='sigmoid'))
model4.add(layers.Flatten()),
model4.add(layers.Dense(64,activation='sigmoid')),
model4.add(layers.Dense(num_classes))

model4.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history4 = model4.fit(train_ds,epochs=5,
                     validation_data=(val_ds))
test_loss4, test_acc4 = model4.evaluate(test_ds,verbose=2)



model6.summary()
plt.plot(history6.history['accuracy'], label='M6accuracy')
plt.plot(history6.history['val_accuracy'], label = 'M6val_accuracy')
plt.plot(history4.history['accuracy'], label='M4accuracy')
plt.plot(history4.history['val_accuracy'], label = 'M4val_accuracy')


plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.95, 1])
plt.xlim([0, 5])
plt.legend(loc='lower right')
plt.savefig("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/PLOTS/M6vsM4BetterData.jpg")
#Now we will use a dataset with 64 labels instead of 10
img_height= 32
img_width= 32
batch_size = 32
train_MT10label=("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/MT10label/Training")
data_dirMT10label=train_MT10label
data_dirMT10label=pathlib.Path(data_dirMT10label)
test_MT10label=("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/MT10label/Test")
datatest_dirMT10label=test_MT10label
datatest_dirMT10label=pathlib.Path(datatest_dirMT10label)
image_countdirMT10labal = len(list(data_dirMT10label.glob("*/*.jpg")))
image_count_dirMT10labeltest=len(list(datatest_dirMT10label.glob("*/*.jpg")))
list_dsMT10label = tf.data.Dataset.list_files(str(data_dirMT10label/'*/*'), shuffle=False)
list_dsMT10label = list_dsMT10label.shuffle(image_countdirMT10labal, reshuffle_each_iteration=False)
class_names = np.array(sorted([item.name for item in data_dirMT10label.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)
#class_names = data_dirMT10label.class_names
#print(class_names)
num_classes=int(len(class_names))
print(num_classes)
AUTOTUNE = tf.data.experimental.AUTOTUNE

list_ds_testMT10label = tf.data.Dataset.list_files(str(datatest_dirMT10label/'*/*'), shuffle=False)
list_ds_testMT10label = list_ds_testMT10label.shuffle(image_count_dirMT10labeltest, reshuffle_each_iteration=False)

val_size = int(image_countdirMT10labal * 0.2)
train_dsMT10label = list_dsMT10label.skip(val_size)
val_dsMT10label = list_dsMT10label.take(val_size)
test_dsMT10label=list_ds_testMT10label.take(int(image_count_dirMT10labeltest))
print(tf.data.experimental.cardinality(train_dsMT10label).numpy())
print(tf.data.experimental.cardinality(val_dsMT10label).numpy())
print(tf.data.experimental.cardinality(test_dsMT10label).numpy())


train_dsMT10label = train_dsMT10label.map(process_path, num_parallel_calls=AUTOTUNE)
val_dsMT10label = val_dsMT10label.map(process_path, num_parallel_calls=AUTOTUNE)

test_dsMT10label = test_dsMT10label.map(process_path, num_parallel_calls=AUTOTUNE)

train_dsMT10label = configure_for_performance(train_dsMT10label)
val_dsMT10label = configure_for_performance(val_dsMT10label)

test_dsMT10label = configure_for_performance(test_dsMT10label)
#Best model so far with more complex and rich data

model6MT10=models.Sequential()
model6MT10.add(layers.Conv2D(64,(3,3),activation='sigmoid',input_shape=(32,32,3))),
model6MT10.add(layers.MaxPooling2D()),
model6MT10.add(layers.Conv2D(64,3,activation='sigmoid')),
model6MT10.add(layers.MaxPooling2D()),
model6MT10.add(layers.Conv2D(64,3,activation='sigmoid')),
#model7MT10.add(layers.Conv2D(64,3,activation='relu')),
model6MT10.add(layers.MaxPooling2D()),
model6MT10.add(layers.Flatten()),
model6MT10.add(layers.Dense(128,activation='sigmoid')),
model6MT10.add(layers.Dense(num_classes))



model6MT10.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

history6MT10 = model6MT10.fit(train_dsMT10label,epochs=10,
                              validation_data=(val_dsMT10label))
test_loss6MT10, test_acc6MT10 = model6MT10.evaluate(test_dsMT10label,verbose=2)


model4MT10=models.Sequential()
#model4.add(layers.Dense(128,activation='relu',input_shape=(32,32,3)))
model4MT10.add(layers.experimental.preprocessing.Rescaling(1. / 255)),
model4MT10.add(layers.Conv2D(32,3,activation='relu', input_shape=(32,32,3))),
model4MT10.add(layers.MaxPooling2D()),
model4MT10.add(layers.Dense(128,activation='relu'))
model4MT10.add(layers.Flatten()),
model4MT10.add(layers.Dense(64,activation='relu')),
model4MT10.add(layers.Dense(num_classes))

model4MT10.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history4MT10 = model4MT10.fit(train_dsMT10label,epochs=10,
                     validation_data=(val_dsMT10label))
test_loss4MT10, test_acc4MT10 = model4MT10.evaluate(test_dsMT10label,verbose=2)

plt.plot(history6MT10.history['accuracy'], label='M6accuracy')
plt.plot(history6MT10.history['val_accuracy'], label = 'M6val_accuracy')
plt.plot(history4MT10.history['accuracy'], label='M4accuracy')
plt.plot(history4MT10.history['val_accuracy'], label = 'M4val_accuracy')


plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.95, 1])
plt.xlim([0, 5])
plt.legend(loc='lower right')
plt.savefig("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/PLOTS/M6vsM4MT10label.jpg")

#Now we will use a dataset with 64 labels instead of 10 (x16)
img_height= 16
img_width= 16
batch_size = 32
train_MT10labelx16=("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/MT10label/Training")
data_dirMT10labelx16=train_MT10labelx16
data_dirMT10labelx16=pathlib.Path(data_dirMT10labelx16)
test_MT10labelx16=("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/MT10label/Test")
datatest_dirMT10labelx16=test_MT10labelx16
datatest_dirMT10labelx16=pathlib.Path(datatest_dirMT10labelx16)
image_countdirMT10labelx16 = len(list(data_dirMT10labelx16.glob("*/*.jpg")))
image_count_dirMT10labeltestx16=len(list(datatest_dirMT10labelx16.glob("*/*.jpg")))
list_dsMT10labelx16 = tf.data.Dataset.list_files(str(data_dirMT10labelx16/'*/*'), shuffle=False)
list_dsMT10labelx16 = list_dsMT10label.shuffle(image_countdirMT10labelx16, reshuffle_each_iteration=False)
class_names = np.array(sorted([item.name for item in data_dirMT10labelx16.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)
#class_names = data_dirMT10label.class_names
#print(class_names)
num_classes=int(len(class_names))
print(num_classes)
AUTOTUNE = tf.data.experimental.AUTOTUNE

list_ds_testMT10labelx16 = tf.data.Dataset.list_files(str(datatest_dirMT10labelx16/'*/*'), shuffle=False)
list_ds_testMT10labelx16 = list_ds_testMT10label.shuffle(image_count_dirMT10labeltestx16, reshuffle_each_iteration=False)

val_size = int(image_countdirMT10labelx16 * 0.2)
train_dsMT10labelx16 = list_dsMT10labelx16.skip(val_size)
val_dsMT10labelx16 = list_dsMT10labelx16.take(val_size)
test_dsMT10labelx16=list_ds_testMT10labelx16.take(int(image_count_dirMT10labeltest))
print(tf.data.experimental.cardinality(train_dsMT10labelx16).numpy())
print(tf.data.experimental.cardinality(val_dsMT10labelx16).numpy())
print(tf.data.experimental.cardinality(test_dsMT10labelx16).numpy())


train_dsMT10labelx16 = train_dsMT10labelx16.map(process_path, num_parallel_calls=AUTOTUNE)
val_dsMT10labelx16 = val_dsMT10labelx16.map(process_path, num_parallel_calls=AUTOTUNE)

test_dsMT10labelx16 = test_dsMT10labelx16.map(process_path, num_parallel_calls=AUTOTUNE)

train_dsMT10labelx16 = configure_for_performance(train_dsMT10labelx16)
val_dsMT10labelx16 = configure_for_performance(val_dsMT10labelx16)

test_dsMT10labelx16 = configure_for_performance(test_dsMT10labelx16)
#Best model so far with more complex and rich data

model6MT10x16=models.Sequential()
model6MT10x16.add(layers.Conv2D(64,(3,3),activation='sigmoid',input_shape=(16,16,3))),
model6MT10x16.add(layers.MaxPooling2D()),
model6MT10x16.add(layers.Conv2D(64,3,activation='sigmoid')),
model6MT10x16.add(layers.MaxPooling2D()),
#model7MT10x16.add(layers.Conv2D(64,3,activation='sigmoid')),
#model7MT10.add(layers.Conv2D(64,3,activation='relu')),
model6MT10x16.add(layers.MaxPooling2D()),
model6MT10x16.add(layers.Flatten()),
model6MT10x16.add(layers.Dense(128,activation='sigmoid')),
model6MT10x16.add(layers.Dense(num_classes))



model6MT10x16.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

history6MT10x16 = model6MT10x16.fit(train_dsMT10labelx16,epochs=10,
                                    validation_data=(val_dsMT10labelx16))
test_loss6x16, test_acc6x16 = model6MT10x16.evaluate(test_dsMT10labelx16,verbose=2)



model4MT10x16=models.Sequential()
model4MT10x16.add(layers.experimental.preprocessing.Rescaling(1. / 255)),
model4MT10x16.add(layers.Conv2D(32,3,activation='relu', input_shape=(16,16,3))),
model4MT10x16.add(layers.MaxPooling2D()),
model4MT10x16.add(layers.Dense(128,activation='relu'))
model4MT10x16.add(layers.Flatten()),
model4MT10x16.add(layers.Dense(64,activation='relu')),
model4MT10x16.add(layers.Dense(num_classes))

model4MT10x16.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history4MT10x16 = model4MT10x16.fit(train_dsMT10label,epochs=10,
                     validation_data=(val_dsMT10label))
test_loss4MT10x16, test_acc4MT10x16 = model4MT10x16.evaluate(test_dsMT10label,verbose=2)






plt.plot(history6MT10x16.history['accuracy'], label='M6accuracy')
plt.plot(history6MT10x16.history['val_accuracy'], label = 'M6val_accuracy')
plt.plot(history4MT10x16.history['accuracy'], label='M4accuracy')
plt.plot(history4MT10x16.history['val_accuracy'], label = 'M4val_accuracy')


plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.95, 1])
#plt.xlim([0, 5])
plt.legend(loc='lower right')
plt.savefig("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/PLOTS/x16M6vsM4MT10label.jpg")


historiesSizer={}

#historiesSizer['8x8'] = x8compile_and_fit(model4_1, "8x8/Small") #model3.1
historiesSizer['16x16']  = compile_and_fitmulti(model6MT10x16,"16x16/Medium",MT10label=True,DIM=(16,16)) #model3.2
historiesSizer['32x32']  = compile_and_fitmulti(model6MT10,"32x32/Large",MT10label=True,DIM=(32,32)) #model3.2
#put models name and principal characteristics
plotter = tfdocs.plots.HistoryPlotter(metric = 'accuracy')
plotter.plot(historiesSizer)
plt.ylim([0.95, 1])

plotter.plot(historiesSizer)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])

plt.savefig("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/PLOTS/M7Battle(x16vsx32).jpg")


plt.plot(history6MT10x16.history['accuracy'], label='accuracy')
plt.plot(history6MT10x16.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.9, 1])
plt.legend(loc='lower right')
plt.savefig("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/PLOTS/MT10.jpg")






def model_magic(X):
summary = str(X.summary())
out = open(X + 'report.txt','w')
out.write(summary)
out.close
 return summary









#More specific label about those 10 (from Apple to Pink lady)
img_height= 32
img_width= 32
batch_size = 32
train_MT10labelSpec=("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/fruits-360/Dataset/Training")
data_dirMT10labelSpec=train_MT10labelSpec
data_dirMT10labelSpec=pathlib.Path(data_dirMT10labelSpec)
test_MT10labelSpec=("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/fruits-360/Dataset/Test")
datatest_dirMT10labelSpec=test_MT10labelSpec
datatest_dirMT10labelSpec=pathlib.Path(datatest_dirMT10labelSpec)
image_countdirMT10labelSpec = len(list(data_dirMT10labelSpec.glob("*/*.jpg")))
image_count_dirMT10labeltestSpec=len(list(datatest_dirMT10labelSpec.glob("*/*.jpg")))
list_dsMT10labelSpec = tf.data.Dataset.list_files(str(data_dirMT10labelSpec/'*/*'), shuffle=False)
list_dsMT10label = list_dsMT10labelSpec.shuffle(image_countdirMT10labelSpec, reshuffle_each_iteration=False)
class_names = np.array(sorted([item.name for item in data_dirMT10labelSpec.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)
#class_names = data_dirMT10label.class_names
#print(class_names)
num_classes=int(len(class_names))
print(num_classes)
AUTOTUNE = tf.data.experimental.AUTOTUNE

list_ds_testMT10labelSpec = tf.data.Dataset.list_files(str(datatest_dirMT10labelSpec/'*/*'), shuffle=False)
list_ds_testMT10labelSpec = list_ds_testMT10labelSpec.shuffle(image_count_dirMT10labeltestSpec, reshuffle_each_iteration=False)

val_size = int(image_countdirMT10labelSpec * 0.2)
train_dsMT10labelSpec = list_dsMT10labelSpec.skip(val_size)
val_dsMT10labelSpec = list_dsMT10labelSpec.take(val_size)
test_dsMT10labelSpec=list_ds_testMT10labelSpec.take(int(image_count_dirMT10labeltestSpec))
print(tf.data.experimental.cardinality(train_dsMT10labelSpec).numpy())
print(tf.data.experimental.cardinality(val_dsMT10labelSpec).numpy())
print(tf.data.experimental.cardinality(test_dsMT10labelSpec).numpy())


train_dsMT10labelSpec = train_dsMT10labelSpec.map(process_path, num_parallel_calls=AUTOTUNE)
val_dsMT10labelSpec = val_dsMT10labelSpec.map(process_path, num_parallel_calls=AUTOTUNE)

test_dsMT10labelSpec = test_dsMT10labelSpec.map(process_path, num_parallel_calls=AUTOTUNE)

train_dsMT10labelSpec = configure_for_performance(train_dsMT10labelSpec)
val_dsMT10labelSpec = configure_for_performance(val_dsMT10labelSpec)

test_dsMT10labelSpec = configure_for_performance(test_dsMT10labelSpec)
#Best model so far with more complex and rich data

data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(random.uniform(0.1,0.9)),
])




resize_and_rescale = tf.keras.Sequential([
  #layers.experimental.preprocessing.Resizing(img_width, img_height),
  layers.experimental.preprocessing.Rescaling(1./255)
])
#batch_size = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

def prepare(ds, shuffle=False, augment=False):
  # Resize and rescale all datasets
  ds = ds.map(lambda x, y: (resize_and_rescale(x), y),
              num_parallel_calls=AUTOTUNE)

  if shuffle:
    ds = ds.shuffle(1000)

  # Batch all datasets
  ds = ds.batch(batch_size)

  # Use data augmentation only on the training set
  if augment:
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=AUTOTUNE)

  # Use buffered prefecting on all datasets
  return ds.prefetch(buffer_size=AUTOTUNE)


def resize_and_rescale(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [img_height, img_width])
  image = (image / 255.0)
  return image, label

def augment(image,label):
  image, label = resize_and_rescale(image, label)
  # Add 6 pixels of padding
  image = tf.image.resize_with_crop_or_pad(image, img_width + 6, img_height + 6)
   # Random crop back to the original size
  image = tf.image.random_crop(image, size=[img_width, img_height, 3])
  image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
  image = tf.clip_by_value(image, 0, 1)
  return image, label


train_dsMT10labelSpec = (
    train_dsMT10labelSpec
    .shuffle(1000)
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

val_dsMT10labelSpec = (
    val_dsMT10labelSpec
    .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

test_dsMT10labelSpec = (
    test_dsMT10labelSpec
    .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)

train_dsMT10labelSpec=prepare(train_dsMT10labelSpec, shuffle=True,augment=True)
#augmented_train_dsMT10labelSpec = train_dsMT10labelSpec.map(
  #lambda x, y: (data_augmentation(x, training=True), y))
#augmented_train_dsMT10labelSpec= augmented_train_dsMT10labelSpec.cache().prefetch(buffer_size=AUTOTUNE)




model7MT10S=models.Sequential()
#model7MT10S.add(layers.Flatten(input_shape=(32,32,3)))
model7MT10S.add(layers.Conv2D(46, (3, 3), activation='sigmoid', input_shape=(32,32,3))),
model7MT10S.add(layers.MaxPooling2D()),
model7MT10S.add(layers.Conv2D(64,3,activation='sigmoid')),
model7MT10S.add(layers.MaxPooling2D()),
model7MT10S.add(layers.Conv2D(128,3,activation='sigmoid')),
#model7MT10.add(layers.Conv2D(64,3,activation='relu')),
model7MT10S.add(layers.MaxPooling2D()),
model7MT10S.add(layers.Flatten()),
model7MT10S.add(layers.Dropout(0.3))
model7MT10S.add(layers.Dense(128,activation='sigmoid')),
#model7MT10S.add(layers.Dense(128,activation='sigmoid')),
#model7MT10S.add(layers.Dense(128,activation='sigmoid')),
model7MT10S.add(layers.Dense(num_classes,activation='softmax'))

#model7MT10S.summary()

model7MT10S.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history7MT10S = model7MT10S.fit(train_dsMT10labelSpec,epochs=5,
                     validation_data=(val_dsMT10labelSpec))
#
history7MT10SLonger = model7MT10S.fit(train_dsMT10labelSpec,epochs=10,
                     validation_data=(val_dsMT10labelSpec))

test_loss7MT10S, test_acc7MT10S = model7MT10S.evaluate(test_dsMT10labelSpec,verbose=2)

plt.plot(history7MT10SLonger.history['accuracy'], label='accuracy')
plt.plot(history7MT10SLonger.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.savefig("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/PLOTS/MT10Adv.jpg")
model7MT10S.save("C:/Users/Gianluca/Desktop/Università/Master UniMi/Project ML&SL/ML/Model/EP200")
#More specific from 64


data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),#RandomRotation(random.uniform(0.1,0.9)),
])



resize_and_rescale = tf.keras.Sequential([
  #layers.experimental.preprocessing.Resizing(img_width, img_height),
  layers.experimental.preprocessing.Rescaling(1./255)
])
image = tf.expand_dims(image, 0)


model7MT10S_aug=tf.keras.Sequential([
   resize_and_rescale,
   data_augmentation,
   layers.Conv2D(300, (3, 3), padding='same', activation='relu', input_shape=(32,32,3)),
   layers.MaxPooling2D(),
   #layers.Conv2D(138,(3, 3),padding='same',activation='relu'),
   #layers.MaxPooling2D(),
   #layers.Conv2D(69,(3, 3),padding='same',activation='relu'),
   #layers.MaxPooling2D(),
   layers.Flatten(),
   layers.Dense(400,activation='relu'),
   #layers.Dropout(0.3),
   layers.Dense(276,activation='relu'),
   layers.Dense(138,activation='relu'),
   layers.Dense(64,activation='relu'),
   layers.Dense(num_classes)
])


model7MT10S_aug.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history7MT10S_aug = model7MT10S_aug.fit(train_dsMT10labelSpec,epochs=10,
                     validation_data=(val_dsMT10labelSpec))
test_loss7MT10S_aug, test_acc7MT10S_aug = model7MT10S_aug.evaluate(test_dsMT10labelSpec,verbose=2)
#
history7MT10SLonger_aug = model7MT10S_aug.fit(train_dsMT10labelSpec,epochs=50,
                     validation_data=(val_dsMT10labelSpec))

tf.keras.backend.clear_session()


test_loss7MT10S_aug, test_acc7MT10S_aug = model7MT10S_aug.evaluate(test_dsMT10labelSpec,verbose=2)
plt.plot(history7MT10SLonger_aug.history['accuracy'], label='accuracy')
plt.plot(history7MT10SLonger_aug.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
