import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

class ConvolutionalNeuralNet():

    def __init__(self):
        self.generator = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            zoom_range=0.10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            horizontal_flip=False,
            fill_mode="nearest"
        )
        self.model = Sequential([
            Conv2D(
                filters=32,  
                kernel_size=(3,3), 
                activation="relu", 
                input_shape=(28,28,1)),
            BatchNormalization(),
            MaxPool2D(2,2, padding='same'),
            Conv2D(filters=128,  kernel_size=(3,3), activation="relu"),
            MaxPool2D(2,2, padding='same'),
            Conv2D(filters=512, kernel_size=(3,3), activation="relu"),
            MaxPool2D(2,2, padding='same'),
            Flatten(),
            Dense(units=1024, activation="relu"),                 
            Dense(units=256, activation="relu"),
            Dropout(0.5),
            Dense(units=25, activation="softmax")
        ])
        self.model.compile(
            optimizer='adam', 
            loss="sparse_categorical_crossentropy", 
            metrics=["accuracy"]
        )
        self.learning_rate_reduction = ReduceLROnPlateau(
            monitor='val_accuracy', 
            patience = 3, 
            verbose=1,
            factor=0.5, 
            min_lr=0.00001
        )

    def fit(self,
            X_train, y_train,
            X_val, y_val,
            batch_size=32,
            epochs=150):
        X_train_flow = self.generator.flow(
            X_train, y_train, 
            batch_size=batch_size
        )
        X_val_flow = self.generator.flow(
            X_val, y_val, 
            batch_size=batch_size
        )
        self.history = self.model.fit(
            X_train_flow, 
            validation_data=X_val_flow, 
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True), 
                self.learning_rate_reduction
            ]
        )

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)