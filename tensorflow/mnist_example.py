import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Reshape
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

x_train = x_train[..., None]
x_test = x_test[..., None]

model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    
    Reshape((7*7, 64)),
    
    LSTM(128, return_sequences=True),
    LSTM(64),
    
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Starting training stress test on GPU...")
start_time = time.time()

model.fit(x_train, y_train, batch_size=128, epochs=3, validation_split=0.1)

end_time = time.time()
print(f"\nTraining completed in {end_time - start_time:.2f} seconds")

loss, acc = model.evaluate(x_test, y_test)
print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
