#loading IMDB dataset
from keras.dataset import imdb
#train_data and test_data: lists of reviews
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

#preparing the data
def vectorize_sequences(sequences, dimension = 10000):
	#create all zero matrix of shape(len(sequences), dimension)
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1
	return results
#vectorized training and testing data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
#ex
x_train[0]

#vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#keras implementation
from keras import models
from keras import layers

model = models.sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000, )))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

#model compiler
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

#if optimizer configuration is desired
from keras import optimizers

model.compile(optimizer = optimizers.RMSprop(lr = 0.01), loss = 'binary_crossentropy', metrics = ['accuracy'])

#if losses and metrics are desired
from keras import losses
from keras import metrics

model.compile(optimizer = optimizers.RMSprop(lr = 0.001), loss = losses.binary_crossentropy, metrics = [metrics.binary_accuracy])

#validation
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#training
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(partial_x_train, partial_y_train, epochs = 20, batch_size = 512, validation_data = (x_val, y_val))

#plotting training / validation loss 
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#plotting training / validation accuracy
plt.clf() #clear fig
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label = 'Training Acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#to avoid overfitting: training model from scratch just for 4 epochs
model = models.sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape(10000, )))
model.add(layer.Dense(16, activation = 'relu'))
model.add(layer.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 4, batch_size = 512)
results = model.evaluate(x_test, y_test)
##end
