#All required libraries are loaded
import cv2
import numpy as np 
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Dropout,BatchNormalization
from tensorflow.keras.applications.resnet_v2 import ResNet50V2,preprocess_input

#Here the number of breeds, image size and batch size is specified
num_breeds = 60
im_size = 224
batch_size = 64
encoder = LabelEncoder()

#The label csv file is read here
df_labels = pd.read_csv(r'''C:/Users/sudhi/Downloads/dog-breed-identification/labels.csv''')
#We store the location of testing and training images folder
train_file = r'''C:/Users/sudhi/Downloads/dog-breed-identification/Dataset Collection/train/'''
test_file = r'''C:/Users/sudhi/Downloads/dog-breed-identification/Dataset Collection/test/'''

#We check the total number of unique breeds in our dataset file
print("Total number of unique Dog Breeds :",len(df_labels.breed.unique()))

#Only 60 unique breed records is obtained
breed_dict = list(df_labels['breed'].value_counts().keys()) 
new_list = sorted(breed_dict,reverse=True)[:num_breeds*2+1:2]
#The dataset is altered so as to only have 60 dogbreeds
df_labels = df_labels.query('breed in @new_list')
#Here we create a new colommn which contains image name with image extension
df_labels['img_file'] = df_labels['id'].apply(lambda x: x + ".jpg")

#A numpy array of the shape is then created
#The function consists of the following - number of dataset records, image size , image size, 3 for rgb channel layer)
#Input of the model
train_x = np.zeros((len(df_labels), im_size, im_size, 3), dtype='float32')

#img_file column of the dataset is iterated
for i, img_id in enumerate(df_labels['img_file']):
  #The image file is read and is converted into numeric format
  #All images are resized to 224x224 pixels
  #Array obtained is (224,224,3) where 3 is the RGB channels layers
  try:
    img = cv2.resize(cv2.imread(train_file+img_id,cv2.IMREAD_COLOR),((im_size,im_size)))
  except:
    break
  #A range of 1 to -1 scale is used on the array
  #The array is preprocessed and its dimensions are expanded on axis 0
  img_array = preprocess_input(np.expand_dims(np.array(img[...,::-1].astype(np.float32)).copy(), axis=0))
  #The train_x variable is updated with new element 
  train_x[i] = img_array

#This is the target for our model
#Breed names are converted into numeric format
train_y = encoder.fit_transform(df_labels["breed"].values)

#The dataset is split in a ratio of 80:20
#80% is used for training while  20% is used for testing purpose
x_train, x_test, y_train, y_test = train_test_split(train_x,train_y,test_size=0.2,random_state=42)

#Image augmentation is created using ImageDataGenerator class
train_datagen = ImageDataGenerator(rotation_range=45,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.25,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

#Images are generated for training sets 
train_generator = train_datagen.flow(x_train, 
                                     y_train, 
                                     batch_size=batch_size)

#Images are generated for testing set the same as above
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow(x_test, 
                                     y_test, 
                                     batch_size=batch_size)

#The model is built with input shape of image array using ResNet50V2
#We obtain the weight for our network using from imagenet dataset
#The first Dense layer is not included
resnet = ResNet50V2(input_shape = [im_size,im_size,3], weights='imagenet', include_top=False)
#All trainable layers are frozen and only top layers are trained 
for layer in resnet.layers:
    layer.trainable = False

#Global average pooling layer and Batch Normalization layer is added
x = resnet.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
#Fully connected layer is added
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

#An output layer is added having the shape equal to number of breeds
predictions = Dense(num_breeds, activation='softmax')(x)

#Model class is created with inputs and outputs
model = Model(inputs=resnet.input, outputs=predictions)

#We add epochs for model training and learning rate for optimizer
epochs = 20
learning_rate = 1e-3

#RMSprop optimizer is used to compile or build the model
optimizer = RMSprop(learning_rate=learning_rate,rho=0.9)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

#The training generator data is fitted and the model is then trained
model.fit(train_generator,
                 steps_per_epoch= x_train.shape[0] // batch_size,
                 epochs= epochs,
                 validation_data= test_generator,
                 validation_steps= x_test.shape[0] // batch_size)

#The model for prediction is saved
model.save("model")

#The model is loaded
model = load_model("model")

#The required image for prediction is loaded
pred_img_path = 'rottweiler.jpg'
#The image file is read and is converted into numeric format
#All images are resized to 224x224 pixels
pred_img_array = cv2.resize(cv2.imread(pred_img_path,cv2.IMREAD_COLOR),((im_size,im_size)))
#A range of 1 to -1 scale is used on the array
#The array is preprocessed and its dimensions are expanded on axis 0
pred_img_array = preprocess_input(np.expand_dims(np.array(pred_img_array[...,::-1].astype(np.float32)).copy(), axis=0))

#The model is fed with the image array for prediction
pred_val = model.predict(np.array(pred_img_array,dtype="float32"))
#The image of the dog is displayed
cv2.imshow("NNFL_Project",cv2.resize(cv2.imread(pred_img_path,cv2.IMREAD_COLOR),((im_size,im_size)))) 
#The predicted breed of the dog is displayed
pred_breed = sorted(new_list)[np.argmax(pred_val)]
print("Predicted Breed for this Dog is :",pred_breed)
