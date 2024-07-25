# #CNN


#importing required libraries
import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau


#mounting google drive
from google.colab import drive
drive.mount('/content/drive')


#importing dataset
dataset = pd.read_csv("/content/drive/MyDrive/dataset_v2.csv") #Path to dataset


#Extracting labels out of the dataset
y = dataset['label']
del dataset['label']


#Extracting the pixels out of the dataset
x = dataset.values



#splitting training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=101)



#using label binarizer to convert labels from 0...n to binary format(0 or 1)
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y1=y_train
y2=y_test
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)


#normalising the data
x_train = x_train / 255
x_test = x_test / 255


#reshaping data to required size
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)


#plotting the pixels to form an image to see how the data looks
f, ax = plt.subplots(2,5) 
f.set_size_inches(10, 10)
k = 0
for i in range(2):
    for j in range(5):
        ax[i,j].imshow(x_train[k].reshape(28, 28) , cmap = "gray")
        k += 1
    plt.tight_layout()   


#transforming data on a random basis
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)


#reduces learning rate when the machine stop learning
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)


#Defining the CNN model 
model = Sequential()

model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Flatten())

model.add(Dense(units = 64 , activation = 'relu'))

model.add(Dropout(0.3))

model.add(Dense(units = 9 , activation = 'softmax'))

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()


#training the defined model with our data
history = model.fit(datagen.flow(x_train,y_train, batch_size = 24) ,epochs = 20 , validation_data = (x_test, y_test) , callbacks = [learning_rate_reduction])


print("Accuracy of the CNN model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")


#sample test 
#18-that's it, 46-yes, 81-okay, 2- i love you, 8- Goodbye, 11- help, 16- no, 5-Thank you, 95- Hello
index=81
plt.imshow(x_test[index].reshape(28,28))


predictions = np.argmax(model.predict(x_test), axis=-1).astype("int32")
for i in range(len(predictions)):
    if(predictions[i] >= 9):
        predictions[i] += 1
x=predictions[index] 
x


f = { 	0:'Hello',
	1:'Yes',
	2:'No',
	3:'Thank you',
	4:'Goodbye',
	5:'I Love You',
	6:'Help',
	7:'Ok',
	8:'Thats it',
}
print(f[x])

 
# ##visualisations


epochs = [i for i in range(20)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(16,9)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Testing Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()


classes = ["Class " + str(i) for i in range(9) if i != 9]
print(classification_report(y2, predictions, target_names=classes))


from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y2, predictions)


cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

 
# #SVM


import numpy as np 
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


dataset = pd.read_csv("/content/drive/MyDrive/dataset_v2.csv")
y = dataset['label']
del dataset['label']
x=dataset.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=101)


label_enc = LabelEncoder()
y_train = label_enc.fit_transform(y_train)
y_test = label_enc.fit_transform(y_test)

from sklearn.svm import SVC

classifier = SVC(decision_function_shape='ovr')

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

acc = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred,average='micro')
cm = confusion_matrix(y_test,y_pred)

print("Confusion Matrix for SVM: ",cm)
print("F1 Score for SVM: ",f1)
print("Accuracy for SVM: ",acc)


print(classification_report(y_test, y_pred, target_names=classes))

 
# #KNN


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

dataset = pd.read_csv("/content/drive/MyDrive/dataset_v2.csv")
y = dataset['label']
del dataset['label']
x=dataset.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=101)

pixel_number = np.arange(0,784,1)

plt.scatter(x_train[0],pixel_number, s=0.4, c = 'r')
plt.scatter(x_train[1],pixel_number, s=0.4, c = 'b')
plt.scatter(x_train[2],pixel_number, s=0.4, c = 'g')
plt.scatter(x_train[3],pixel_number, s=0.4, c = 'y')
plt.scatter(x_train[4],pixel_number, s=0.4, c = 'm')
plt.show()

label_enc = LabelEncoder()
y_train = label_enc.fit_transform(y_train)
y_test = label_enc.fit_transform(y_test)

from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors=16)
classifier = KNN.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
acc = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred,average='micro')
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix for RNN: ")
print(cm)
print("F1 score of KNN model: ",f1)
print("Accuracy of KNN model: ",acc)


print(classification_report(y_test, y_pred, target_names=classes))

 
# #RealTimeObjectDetection


from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename


import tensorflow as tf
import tensorflow_hub as hub
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']


def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img


def run_detector(detector, path):
  img = load_img(path)

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  result = detector(converted_img)

  result = {key:value.numpy() for key,value in result.items()}
  hand_index = 0
  for i in range(len(result['detection_class_entities'])):
    img_class = result['detection_class_entities'][i].decode("utf-8")
    if img_class == 'Human hand':
        hand_index = i
        break
  CROP_SIZE = (28,28)
  img = img[None, ...]
  output = tf.image.crop_and_resize(img, [result['detection_boxes'][hand_index]], [0], CROP_SIZE)
  output[0].shape
  tf.keras.utils.save_img(
    'cropped.jpg', output[0], data_format=None, file_format=None, scale=True
  )
  return result




from gtts import gTTS
from IPython.display import Audio
from IPython.display import display


from IPython.display import Image
try:
  filename = take_photo()
  print('Saved to {}'.format(filename))
  
  # Show the image which was just taken.
  display(Image(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))


result = run_detector(detector, "/content/photo.jpg")


from PIL import Image
import numpy
from numpy import asarray

img = Image.open("/content/cropped.jpg")  
imgGray = img.convert('L')
gray_image_data = asarray(imgGray).flatten()
gray_image_data = gray_image_data.tolist()
gray_image_data = [gray_image_data]
print(gray_image_data)


array = numpy.array(gray_image_data)
array = array/255
array = array.reshape(-1,28,28,1)
predictions = np.argmax(model.predict(array), axis=-1)
f = { 
        0:'Hello',
        1:'Yes',
        2:'No',
        3:'Thank you',
        4:'Goodbye',
        5:'I Love You',
     
     
        6:'Help',
        7:'Ok',
        8:'Thats it',
}
predictions[0]


print(f[predictions[0]])


mytext = f[predictions[0]]
audio = gTTS(text=mytext, lang="en", slow=False)
audio.save("example.mp3")


sound_file = '/content/example.mp3'
wn = Audio(sound_file, autoplay=True) 
display(wn)

 
# #Gradio
# 


import numpy
from PIL import Image
import csv
from numpy import asarray
import tensorflow as tf
import tensorflow_hub as hub


def HandSign(Image):
    img = Image.open(Image)

    img = load_img(path)

    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)

    result = {key:value.numpy() for key,value in result.items()}
    hand_index = 0
    for i in range(len(result['detection_class_entities'])):
      img_class = result['detection_class_entities'][i].decode("utf-8")
      if img_class == 'Human hand':
          hand_index = i
          break
    CROP_SIZE = (28,28)
    img = img[None, ...]
    output = tf.image.crop_and_resize(img, [result['detection_boxes'][hand_index]], [0], CROP_SIZE)
    output[0].shape
    tf.keras.utils.save_img(
      'cropped.jpg', output[0], data_format=None, file_format=None, scale=True
    )

    img = Image.open("/content/cropped.jpg")  
    imgGray = img.convert('L')
    gray_image_data = asarray(imgGray).flatten()
    gray_image_data = gray_image_data.tolist()
    gray_image_data = [gray_image_data]
    
    array = numpy.array(gray_image_data)
    array = array/255
    array = array.reshape(-1,28,28,1)
    predictions = np.argmax(model.predict(array), axis=-1)
    f = { 
            0:'Hello',
            1:'Yes',
            2:'No',
            3:'Thank you',
            4:'Goodbye',
            5:'I Love You',
            6:'Help',
            7:'Ok',
            8:'Thats it',
    }
    return f[predictions[0]]

import gradio as gr
demo = gr.Interface(
    fn=HandSign,
    inputs=gr.Image(source="webcam"), 
    outputs="text"
)
demo.launch()