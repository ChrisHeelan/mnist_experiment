import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from base_dockex.BaseKerasModel import BaseKerasModel


class KerasCNN(BaseKerasModel):
    def __init__(self, input_args):
        super().__init__(input_args)

        self.first_cnn_units = self.params['first_cnn_units']
        self.second_cnn_units = self.params['second_cnn_units']
        self.dense_units = self.params['dense_units']
        self.patience = self.params['patience']

    def instantiate_model(self):
        print('Instantiating model')

        input_shape = self.X_train.shape[1:]
        num_classes = self.y_train.shape[1]

        self.model = Sequential()
        self.model.add(Conv2D(self.first_cnn_units, kernel_size=(3, 3),
                       activation='relu',
                       input_shape=input_shape))
        self.model.add(Conv2D(self.second_cnn_units, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(self.dense_units, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adadelta())

        self.callbacks.append(EarlyStopping(
            patience=self.patience,
            restore_best_weights=True,
            verbose=2
        ))

        print(self.model.summary())


if __name__ == '__main__':
    print(sys.argv)
    KerasCNN(sys.argv).run()
