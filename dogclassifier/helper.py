from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense
import cv2
import numpy as np
from .constants import DOG_NAMES           

class DogClassifier():

    def __init__(self, image_path):
        self.image_path = image_path
        self.ResNet50_model = ResNet50(weights='imagenet')

    def dog_classifer(self):
        if self.dog_detector():
            print("Dog is detected")
            self.ResNet50_model = Sequential()
            self.ResNet50_model.add(GlobalAveragePooling2D(input_shape=(1,1,2048)))
            self.ResNet50_model.add(Dense(133, activation='softmax'))
            print(self.ResNet50_model.summary())
            self.ResNet50_model.load_weights('dogclassifier/saved_models/weights.best.Resnet50.hdf5')
            return ['dog', self.Resnet50_predict_breed().split('.')[-1]]
        elif self.face_detector():
            print("Human face is detected")
            self.ResNet50_model = Sequential()
            self.ResNet50_model.add(GlobalAveragePooling2D(input_shape=(1,1,2048)))
            self.ResNet50_model.add(Dense(133, activation='softmax'))
            print(self.ResNet50_model.summary())
            self.ResNet50_model.load_weights('dogclassifier/saved_models/weights.best.Resnet50.hdf5')
            return ['human', self.Resnet50_predict_breed().split('.')[-1]]
        else:
            raise ValueError("Dog or Face not detected")


    def Resnet50_predict_breed(self):
        # extract bottleneck features
        bottleneck_feature = self.extract_Resnet50(self.path_to_tensor())
        print(bottleneck_feature.shape) #returns (1, 2048)
        bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
        bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
        print(bottleneck_feature.shape)
        # obtain predicted vector
        predicted_vector = self.ResNet50_model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        # load list of dog names
        return DOG_NAMES[np.argmax(predicted_vector)]

    def dog_detector(self):
        prediction = self.ResNet50_predict_labels()
        return ((prediction <= 268) & (prediction >= 151))

    def face_detector(self):
        face_cascade = cv2.CascadeClassifier('dogclassifier/haarcascades/haarcascade_frontalface_alt.xml')
        img = cv2.imread(self.image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def ResNet50_predict_labels(self):
        # returns prediction vector for image located at img_path
        img = preprocess_input(self.path_to_tensor())
        return np.argmax(self.ResNet50_model.predict(img))

    def path_to_tensor(self):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(self.image_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def extract_Resnet50(self, tensor):
        return ResNet50(weights='imagenet', include_top=False, pooling="avg").predict(preprocess_input(tensor))