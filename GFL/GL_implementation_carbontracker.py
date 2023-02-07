import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from matplotlib import pyplot as plt
import random
import os, sys
from datetime import datetime 
import time
import collections
import tensorflow_datasets as tfds
from carbontracker.tracker import CarbonTracker


#deterministic behaviour
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

#Parameters
TEST_CLIENTS = 3183
NUM_CLIENTS = 200
NUM_EPOCHS = 5  
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
NUM_ROUNDS = 200
MERGE_STEP = True
MODEL = "CNN"
path = f"EMNIST_performance_{MODEL}/NUM_TEST_CLIENTS_{NUM_CLIENTS}_NUM_EPOCHS_{NUM_EPOCHS}_BATCH_SIZE_{BATCH_SIZE}_NUM_ROUNDS_{NUM_ROUNDS}/"

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data() #Download the mnist version with only digits 
emnist_eval, emnist_test = tff.simulation.datasets.ClientData.train_test_client_split(emnist_test, TEST_CLIENTS, seed=0)

#create the validation dataset 
x_val = []
y_val = []
emnist_eval_all_clients = emnist_eval.create_tf_dataset_from_all_clients(seed=0)
emnist_eval_all_clients = tfds.as_numpy(emnist_eval_all_clients)
for sample in emnist_eval_all_clients:
    if MODEL == "FFNN":
        x_val.append(sample["pixels"].reshape(784))
        y_val.append(sample["label"].reshape(1))
    elif MODEL == "CNN":
        x_val.append(sample["pixels"].reshape((28,28,1)))
        y_val.append(sample["label"].reshape(1))

y_val = np.array(y_val)
x_val = np.array(x_val)

if MODEL == "FFNN":
    net_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
            tf.keras.layers.Dense(200, activation = 'relu', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=SEED)),
            tf.keras.layers.Dense(200, activation = 'relu', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=SEED)),
            tf.keras.layers.Dense(10, kernel_initializer = tf.keras.initializers.glorot_uniform(seed=SEED)),
            tf.keras.layers.Softmax(), #activation of the previous dense layer 
    ])
elif MODEL == "CNN":
    net_model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28,28,1)),
            tf.keras.layers.Conv2D(32, (5, 5), kernel_initializer=tf.keras.initializers.glorot_uniform(seed=SEED)),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (5, 5), kernel_initializer=tf.keras.initializers.glorot_uniform(seed=SEED)),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu",
                                  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=SEED)),
            tf.keras.layers.Dense(10, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=SEED)),
            tf.keras.layers.Softmax(),
        ])

net_model.compile(
    optimizer =tf.keras.optimizers.SGD(learning_rate = 0.02),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])

#print(net_model.get_weights())

training_accuracy_log = []
training_loss_log = []
training_accuracy_log_matrix = []
training_loss_log_matrix = []
validation_accuracy_log  = []
validation_loss_log = []
round_time_log = []
local_time_log = []
local_time_log_matrix = []

#initialized the last received model for each client 
last_received_model = {}
initialized_weights = [layer.get_weights() for layer in net_model.layers]
for client_id in emnist_train.client_ids:
    last_received_model[client_id] = initialized_weights.copy()
'''
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(initialized_weights[0])
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
'''
'''
for client_id in emnist_train.client_ids:
    last_received_model["client_id"] = net_model.get_weights()
ls'''
tracker = CarbonTracker(epochs=NUM_ROUNDS, epochs_before_pred = -1, monitor_epochs = NUM_ROUNDS, log_dir=path, update_interval=2, devices_by_pid=True)

for round_num in range(0, NUM_ROUNDS):
    tracker.epoch_start()
    round_time_start = time.time()

    sample_clients = random.sample(emnist_train.client_ids, k = NUM_CLIENTS)
    random.shuffle(sample_clients)
    
    local_time_log_iteration = []
    for client_id in sample_clients:
        client_train_data = emnist_train.create_tf_dataset_for_client(client_id)
        client_train_data_np = tfds.as_numpy(client_train_data)
        x_train = []
        y_train = []
        for sample in client_train_data_np:
            if MODEL == "FFNN":
                x_train.append(sample['pixels'].reshape(784))
                y_train.append(sample['label'].reshape(1))
            elif MODEL == "CNN":
                x_train.append(sample['pixels'].reshape((28,28,1)))
                y_train.append(sample['label'].reshape(1))
        
        x_train = np.array(x_train)
        y_train = np.array(y_train)
    
        local_time_start = time.time()

        if MERGE_STEP:
            #perform the merge operation 
            last_model_weights = last_received_model[client_id].copy()
            current_model_weights = [layer.get_weights() for layer in net_model.layers]
            for layer, last, current in zip(net_model.layers, last_model_weights, current_model_weights):
                if len(last) == 0: #the softmax layer has no weights 
                    continue
                #print(last)
                avg_w = np.mean(np.array([last[0], current[0]]), axis = 0)
                avg_b = np.mean(np.array([last[1], current[1]]), axis = 0)
                layer.set_weights([avg_w, avg_b])
            
            '''
            debug_model_weights_after_merge = [layer.get_weights() for layer in net_model.layers]
            print("///////////////////////////////////////////////////////")
            print(debug_model_weights_after_merge[0])
            print("///////////////////////////////////////////////////////")
            '''
            last_received_model[client_id] = current_model_weights.copy()
        #model training 
        hist_train = net_model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs= NUM_EPOCHS, verbose = 0)
        local_time_end = time.time()
        local_time = local_time_end - local_time_start
        local_time_log_iteration.append(local_time)
        training_accuracy_log_matrix.append(hist_train.history["sparse_categorical_accuracy"])
        training_loss_log_matrix.append(hist_train.history["loss"])

    #model on the validation set
    val_metrics = net_model.evaluate(x = x_val, y = y_val, return_dict = True, verbose = 0)

    #get model performance
    training_accuracy = hist_train.history['sparse_categorical_accuracy'][NUM_EPOCHS-1]
    training_loss = hist_train.history['loss'][NUM_EPOCHS-1]
    validation_accuracy =val_metrics['sparse_categorical_accuracy']
    validation_loss = val_metrics['loss']

    #log model performance 
    training_loss_log.append(training_loss)
    training_accuracy_log.append(training_accuracy)
    validation_loss_log.append(validation_loss)
    validation_accuracy_log.append(validation_accuracy)
    
    local_time_log_matrix.append(local_time_log_iteration) #just to log the local time of each node for each iteration 
    local_time_log.append(np.sum(local_time_log_iteration)) #compute the sum of the local time for each iteration as the sum of local time of each node
    round_time_end = time.time()
    round_time = round_time_end - round_time_start
    round_time_log.append(round_time)

    print(f"Round {round_num}, training accuracy: {training_accuracy}, validation accuracy {validation_accuracy}, Local iteration time: {np.sum(local_time_log_iteration)}, iteration time: {round_time}")
    tracker.epoch_end()

tracker.stop()
os.makedirs(path, exist_ok=True)

plt.figure()
plt.title("Accuracy")
plt.plot(range(0,NUM_ROUNDS), training_accuracy_log, label="Training accuracy")
plt.plot(range(0,NUM_ROUNDS), validation_accuracy_log, label = "Validation accuracy")
plt.grid()
plt.legend()
plt.savefig(path + "performance.jpg")

plt.figure()
plt.title("Execution time")
plt.plot(range(0, NUM_ROUNDS), round_time_log, label = "Round time")
plt.plot(range(0, NUM_ROUNDS), local_time_log, label = "Local iteration time")
plt.legend()
plt.grid()
plt.savefig(path + "time.jpg")

#save the model
net_model.save(path + "model.h5")

#test set usage
x_test = []
y_test = []

emnist_test_all_clients = emnist_test.create_tf_dataset_from_all_clients(seed=0)
emnist_test_all_clients = tfds.as_numpy(emnist_test_all_clients)
for sample in emnist_test_all_clients:
    if MODEL == "FFNN":
        x_test.append(sample["pixels"].reshape(784))
        y_test.append(sample["label"].reshape(1))
    elif MODEL == "CNN":
        x_test.append(sample['pixels'].reshape((28,28,1)))
        y_test.append(sample["label"].reshape(1))
x_test = np.array(x_test)
y_test = np.array(y_test)
test_performance = net_model.evaluate(x=x_test, y=y_test, return_dict = True, verbose = 0)
print(f"Test set accuracy: {test_performance['sparse_categorical_accuracy']}")

np.save(path+"training_accuracy", np.array(training_accuracy_log))
np.save(path+"training_loss", np.array(training_loss_log))
np.save(path+"validation_accuracy", np.array(validation_accuracy_log))
np.save(path+"validation_loss", np.array(validation_loss_log)) 
np.save(path+"local_time", np.array(local_time_log))
np.save(path+"local_time_matrix", np.array(local_time_log_matrix))
np.save(path+"round_time", np.array(round_time_log))
np.save(path+"test_loss", np.array(test_performance["loss"]))
np.save(path+"test_accuracy", np.array(test_performance["sparse_categorical_accuracy"]))
