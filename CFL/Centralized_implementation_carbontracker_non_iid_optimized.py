#https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification

import collections
from genericpath import exists
import numpy as np
import tensorflow as tf 
import tensorflow_federated as tff 
from matplotlib import pyplot as plt 
import random 
import pickle
import tqdm 
import os
from datetime import datetime
import time
import tensorflow_datasets as tfds
from carbontracker.tracker import CarbonTracker

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#deterministic behaviour
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
#tf.config.experimental.enable_op_determinism()

#Parameters
TEST_CLIENTS = 3183
NUM_CLIENTS = 200
NUM_EPOCHS = 5 #number of epochs done locally before the communication step 
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
NUM_ROUNDS = 200
NUM_CLASS_FOR_USER = 3
#path = f"/home/eguerra/Centralized/EMNIST_performance_definitive/{datetime.now()}_NUM_TEST_CLIENTS_{NUM_CLIENTS}_NUM_EPOCHS_{NUM_EPOCHS}_BATCH_SIZE_{BATCH_SIZE}_NUM_ROUNDS_{NUM_ROUNDS}/"
MODEL = "CNN"
path = f"EMNIST_{MODEL}_non_iid/NUM_TEST_CLIENTS_{NUM_CLIENTS}_NUM_EPOCHS_{NUM_EPOCHS}_BATCH_SIZE_{BATCH_SIZE}_NUM_ROUNDS_{NUM_ROUNDS}_MODEL_{MODEL}/"

show_figure = False
os.makedirs(path)

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data() #Download the mnist version with only digits 
emnist_eval, emnist_test = tff.simulation.datasets.ClientData.train_test_client_split(emnist_test, TEST_CLIENTS, seed=SEED)

client_data_dict = {}
for id in emnist_train.client_ids:
    classes_set = np.random.choice(range(0,10), NUM_CLASS_FOR_USER, replace = False)
    client_dataset = emnist_train.create_tf_dataset_for_client(id)

    labels = []
    pixels = []
    for sample in client_dataset:
        if sample["label"].numpy() in classes_set:
            pixels.append(sample["pixels"])
            labels.append(sample["label"])
    updated_dataset = tf.data.Dataset.from_tensor_slices({"label": labels, "pixels": pixels})
    if len(labels) != len(pixels):
        print("ERROR DIM!!!!!!!")
    client_data_dict[id] = updated_dataset

def create_tf_dataset_for_client_fn(client_id):
    return client_data_dict[client_id]

emnist_train_pruned = tff.simulation.datasets.ClientData.from_clients_and_fn(client_ids = emnist_train.client_ids, create_tf_dataset_for_client_fn= create_tf_dataset_for_client_fn)

f = plt.figure(figsize=(12,7))
f.suptitle("Label Counts for a sample of clients")
for i in range(6):
    client_dataset = emnist_train_pruned.create_tf_dataset_for_client(emnist_train_pruned.client_ids[i])

    plot_data = collections.defaultdict(list)
    for example in client_dataset:
        label = example['label'].numpy()
        plot_data[label].append(label)
    plt.subplot(2,3,i+1)
    plt.title("Client {}".format(i))
    for  j in range(10):
        plt.hist(plot_data[j], density = False, bins = [0,1,2,3,4,5,6,7,8,9,10])

plt.savefig(path+"dataset_distribution.jpg")

def preprocess(dataset):
    def batch_format_fn(element):

        if MODEL == "FFNN":
            # Flatten a batch pixels and return the fatures as an 'OrderedDict'
            prep_dataset = collections.OrderedDict(
                x=tf.reshape(element['pixels'], [-1, 784]),
                y=tf.reshape(element['label'], [-1, 1])
            )
        elif MODEL == "CNN":
            prep_dataset = collections.OrderedDict(
                x=tf.reshape(element['pixels'], [-1, 28, 28, 1]),
                y=tf.reshape(element['label'], [-1, 1])
            )
        return prep_dataset
    
    repeated_dataset = dataset.repeat(NUM_EPOCHS) #repeats the dataset for a specific number of times 
    shuffled_dataset = repeated_dataset.shuffle(SHUFFLE_BUFFER, seed=SEED) #randomply shuffles the element of the dataset The dataset fills a buffer with buffer_size elements then randomly samples eleemtns from this buffer replacing selected elements with new elements
    batched_dataset = shuffled_dataset.batch(BATCH_SIZE) #combines consecutive elements of this dataset into batches 
    map_dataset = batched_dataset.map(batch_format_fn) #applies the function to each element of the dataset 
    prefetch_dataset = map_dataset.prefetch(PREFETCH_BUFFER) #allows later elements to be prepared while the current element is being processe. Improves latency and throughput at the cost of using additional memory to store prefetched elements 
    return prefetch_dataset

def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]


### Creating a model with Keras ###
def create_keras_model():
    if MODEL == "FFNN":
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(784,)),
            tf.keras.layers.Dense(200, activation = 'relu', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=SEED)),
            tf.keras.layers.Dense(200, activation = 'relu', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=SEED)),
            tf.keras.layers.Dense(10, kernel_initializer = tf.keras.initializers.glorot_uniform(seed=SEED)),
            tf.keras.layers.Softmax(), #activation of the previous dense layer
        ])
    elif MODEL == "CNN":
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (5, 5), kernel_initializer=tf.keras.initializers.glorot_uniform(seed=SEED)),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (5, 5), kernel_initializer=tf.keras.initializers.glorot_uniform(seed=SEED)),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu",
                                  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=SEED)),
            tf.keras.layers.Dense(10, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=SEED)),
            tf.keras.layers.Softmax()
        ])
    return model
example_dataset = emnist_train_pruned.create_tf_dataset_for_client(emnist_train_pruned.client_ids[0])
preprocessed_example_dataset = preprocess(example_dataset)
#print(preprocessed_example_dataset.element_spec)

#to use any model with TFF must be wrapped in tff.learning.Model 
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec = preprocessed_example_dataset.element_spec, 
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])

#There are two optimizers one at the client side and another one for the server side 
iterative_process = tff.learning.build_federated_averaging_process(model_fn, 
                    client_optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate = 0.02), 
                    server_optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate = 1.0))

#logdir = "/home/elia/Python/TFF/logdir_d_initialize_300clients_10batch"
#summary_writer = tf.summary.create_file_writer(logdir)
state = iterative_process.initialize() #initialization of the server model  
federated_evaluation = tff.learning.build_federated_evaluation(model_fn)
federated_evaluation_data = make_federated_data(emnist_eval, emnist_eval.client_ids)
training_accuracy_log = []
training_loss_log = []
validation_accuracy_log  = []
validation_loss_log = []
round_time_log = []
local_time_log = []

tracker = CarbonTracker(epochs=NUM_ROUNDS, epochs_before_pred=-1, monitor_epochs=NUM_ROUNDS, log_dir = path, update_interval=2, devices_by_pid=True)

#with summary_writer.as_default():
for round_num in range(0, NUM_ROUNDS): #here we should change the set of elements that we are taking into account 
    tracker.epoch_start()
    round_time_start = time.time()
    #At each round we have to select a different subset of devices 
    #For semplicity at each iteration we use the same clients 
    
    sample_clients = random.sample(emnist_train_pruned.client_ids, k=NUM_CLIENTS)
    federated_train_data = make_federated_data(emnist_train_pruned, sample_clients)
    
    #print("Number of client datasets: {l}".format(l=len(federated_train_data)))
    #print("First dataset: {d}".format(d=federated_train_data[0]))
    local_time_start = time.time()
    state , train_metrics = iterative_process.next(state, federated_train_data)
    local_time_end = time.time()
    local_time = local_time_end - local_time_start
    local_time_log.append(local_time)
    validation_metrics = federated_evaluation(state.model, federated_evaluation_data)
    
    #get model performance 
    training_accuracy = train_metrics["train"]["sparse_categorical_accuracy"]
    training_loss = train_metrics["train"]["loss"]
    validation_accuracy = validation_metrics["sparse_categorical_accuracy"]
    validation_loss = validation_metrics["loss"]
    
    #log model performance
    training_accuracy_log.append(training_accuracy)
    training_loss_log.append(training_loss)
    validation_accuracy_log.append(validation_accuracy)
    validation_loss_log.append(validation_loss)

    round_time_end = time.time()
    round_time = round_time_end - round_time_start
    round_time_log.append(round_time)
    print(f"Round {round_num}, training accuracy: {training_accuracy}, validation accuracy {validation_accuracy}, training time: {local_time}, iteration time: {round_time}")
    tracker.epoch_end()

tracker.stop()
#save results


plt.figure()
plt.title("Accuracy")
plt.plot(range(0, NUM_ROUNDS), training_accuracy_log, label = "Training Accuracy")
plt.plot(range(0, NUM_ROUNDS), validation_accuracy_log, label = "Validation accuracy")
plt.legend()
plt.grid()
plt.savefig(path + "performance.jpg")

plt.figure()
plt.title ("Execution time")
plt.plot(range(0,NUM_ROUNDS), round_time_log, label = "Round time")
plt.plot(range(0, NUM_ROUNDS), local_time_log, label = "Local Iteration time")
plt.legend()
plt.grid()
plt.savefig(path + "time.jpg")

#save the model
keras_model = create_keras_model()
state.model.assign_weights_to(keras_model)
keras_model.save(path + "model.h5")
keras_model = tf.keras.models.load_model(path + "model.h5")
keras_model.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate = 0.02),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    )

#validation set check
x_eval = []
y_eval = []
emnist_eval_all_clients = emnist_eval.create_tf_dataset_from_all_clients(seed=SEED)
emnist_eval_all_clients = tfds.as_numpy(emnist_eval_all_clients)
for sample in emnist_eval_all_clients:
    if MODEL == "FFNN":
        x_eval.append(sample["pixels"].reshape(784))
        y_eval.append(sample["label"].reshape(1))
    elif MODEL == "CNN":
        x_eval.append(sample["pixels"].reshape((28,28,1)))
        y_eval.append(sample["label"].reshape(1))

x_eval = np.array(x_eval)
y_eval = np.array(y_eval)

keras_eval_dict = keras_model.evaluate(x=x_eval, y=y_eval, return_dict = True, verbose = 0)
print(f"Validation accuracy keras model: {keras_eval_dict['sparse_categorical_accuracy']}")

#test set usage 
x_test = []
y_test = []

emnist_test_all_clients = emnist_test.create_tf_dataset_from_all_clients(seed=SEED)
emnist_test_all_clients = tfds.as_numpy(emnist_test_all_clients)
for sample in emnist_test_all_clients:
    if MODEL == "FFNN":
        x_test.append(sample["pixels"].reshape(784))
        y_test.append(sample["label"].reshape(1))
    elif MODEL == "CNN":
        x_test.append(sample["pixels"].reshape((28, 28, 1)))
        y_test.append(sample["label"].reshape(1))


x_test = np.array(x_test)
y_test = np.array(y_test)
test_performance = keras_model.evaluate(x=x_test, y=y_test, return_dict = True, verbose = 0)
print(f"Test set accuracy: {test_performance['sparse_categorical_accuracy']}")

np.save(path+"training_accuracy", np.array(training_accuracy_log))
np.save(path+"training_loss", np.array(training_loss_log))
np.save(path+"validation_accuracy", np.array(validation_accuracy_log))
np.save(path+"validation_loss", np.array(validation_loss_log)) 
np.save(path+"local_time", np.array(local_time_log))
np.save(path+"round_time", np.array(round_time_log))
np.save(path+"test_loss", np.array(test_performance["loss"]))
np.save(path+"test_accuracy", np.array(test_performance["sparse_categorical_accuracy"]))
