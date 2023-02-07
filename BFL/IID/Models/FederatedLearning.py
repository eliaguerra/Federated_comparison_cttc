from InputsConfig_FLchain import InputsConfig as p
from InputsConfig_FLchain import InputsConfig as p
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import random
import os
import time

tf.random.set_seed(0)
os.environ['PYTHONHASHSEED'] = str(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
random.seed(0)

class FederatedLearning():
    """ Defines the Federated Learning operation.

        :param str type:
    """

    # Pre-processing function
    def preprocess(self, dataset):

        def batch_format_fn(element):
            if p.MODEL == "FFNN":
                # Flatten a batch pixels and return the fatures as an 'OrderedDict'
                prep_dataset = collections.OrderedDict(
                    x=tf.reshape(element['pixels'], [-1, 784]),
                    y=tf.reshape(element['label'], [-1, 1])
                )
            elif p.MODEL == "CNN":
                prep_dataset = collections.OrderedDict(
                    x=tf.reshape(element['pixels'], [-1, 28, 28, 1]),
                    y=tf.reshape(element['label'], [-1, 1])
                )

            return prep_dataset

        return dataset.repeat(p.NUM_EPOCHS).shuffle(p.SHUFFLE_BUFFER, seed=0).batch(
            p.BATCH_SIZE).map(batch_format_fn).prefetch(p.PREFETCH_BUFFER)

    # Make data federated
    def make_federated_data(self, client_data, client_ids):
        return [
            self.preprocess(client_data.create_tf_dataset_for_client(x))
            for x in client_ids
        ]

    # Define the NN model to be used
    def create_keras_model(self):
        if p.MODEL == "FFNN":
            model = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(784,)),
                tf.keras.layers.Dense(200, activation = 'relu', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=p.SEED)),
                tf.keras.layers.Dense(200, activation = 'relu', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=p.SEED)),
                tf.keras.layers.Dense(10, kernel_initializer = tf.keras.initializers.glorot_uniform(seed=p.SEED)),
                tf.keras.layers.Softmax(),
            ])
        elif p.MODEL == "CNN":
            model = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(28,28,1)),
                tf.keras.layers.Conv2D(32, (5, 5), kernel_initializer=tf.keras.initializers.glorot_uniform(seed=p.SEED)),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(64, (5, 5), kernel_initializer=tf.keras.initializers.glorot_uniform(seed=p.SEED)),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="relu",
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=p.SEED)),
                tf.keras.layers.Dense(10, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=p.SEED)),
                tf.keras.layers.Softmax()
            ])
        return model

    def create_tf_dataset_for_client_fn(client_id):
        # Get the original client's dataset to be modified
        client_dataset_copy = p.emnist_train.create_tf_dataset_for_client(
            p.emnist_train.client_ids[client_id])
        # Choose random classes to remain
        classes_set = np.random.choice(range(0, 10), p.NUM_CLASSES_PER_USER, replace=False)
        # List to store the valid samples
        elements = []
        # Iterate for each element in the original client's dataset
        for sample in client_dataset_copy:
            # Select only the samples matching with classes_set
            if sample['label'].numpy() in classes_set:
                elements.append({'label': sample['label'], 'pixels': sample['pixels']})
        # Generate the dataset object for this specific cient
        updated_dataset = tf.data.Dataset.from_generator(
            lambda: elements, {"label": tf.int32, "pixels": tf.float32})
        # Return the dataset
        return updated_dataset

    # Model constructor (needed to be passed to TFF, instead of a model instance)
    def model_fn(self):

        keras_model = self.create_keras_model()

        # Define the ML model compliant with FL (needs a sample of the dataset to be defined)
        sample_dataset = p.emnist_train.create_tf_dataset_for_client(p.emnist_train.client_ids[0])
        preprocessed_sample = self.preprocess(sample_dataset)

        return tff.learning.from_keras_model(
            keras_model,
            input_spec=preprocessed_sample.element_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(),
                     tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        # More loss functions and metrics here:
        # - https://www.tensorflow.org/api_docs/python/tf/keras/losses
        # - https://www.tensorflow.org/api_docs/python/tf/keras/metrics

    # Initialize FL model
    def initialize_fl_model(self):

        iterative_process = tff.learning.build_federated_averaging_process(
            self.model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=p.LEARNING_RATE_CLIENT),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=p.LEARNING_RATE_SERVER))

        state = iterative_process.initialize()

        fed_evaluation = tff.learning.build_federated_evaluation(self.model_fn)
        federated_evaluation_data = self.make_federated_data(p.emnist_eval, p.emnist_eval.client_ids)
        return iterative_process, state, fed_evaluation, federated_evaluation_data

    # Execute an FL round
    def execute_fl_round(self, iterative_process, state, fed_evaluation, federated_evaluation_data, transactions):
        #round_time_start = time.time()
        # Get participating clients' dataset and make it federated
        
        participating_clients_ids = []
        for i in transactions:
            participating_clients_ids.append(p.emnist_train.client_ids[i.sender-1])

        #sample_clients = random.choices(p.emnist_train.client_ids, k=p.NUM_CLIENTS)
        # Make data from participating clients federated
        federated_train_data = self.make_federated_data(p.emnist_train, participating_clients_ids)
        # Execute a training round
        local_time_start = time.time()
        state, train_metrics = iterative_process.next(state, federated_train_data)
        local_time_end = time.time()
        local_time = local_time_end - local_time_start
        # Get validation metrics
        validation_metrics = fed_evaluation(state.model, federated_evaluation_data)
        # Get test metrics --> to do
        #test_metrics = 
        
        #round_time_end = time.time()
        #round_time = round_time_end - round_time_start 
        '''
        # Choose another set of random clients for evaluation
        sample_random_clients_ids = random.sample(range(0, len(p.emnist_test.client_ids) - 1), p.NUM_CLIENTS_TEST)
        sample_random_clients = []
        for idx in sample_random_clients_ids:
            sample_random_clients.append(p.emnist_test.client_ids[idx])
        eval_datasets = self.make_federated_data(p.emnist_test, sample_random_clients)
        eval_metrics = fed_evaluation(iterative_process.get_model_weights(state), eval_datasets)
        '''
        return state, train_metrics, validation_metrics, local_time

    # @tf.function
    # def client_update(model, dataset, server_weights, client_optimizer):
    #     """Performs training (using the server model weights) on the client's dataset."""
    #     # Initialize the client model with the current server weights.
    #     client_weights = model.trainable_variables
    #     # Assign the server weights to the client model.
    #     tf.nest.map_structure(lambda x, y: x.assign(y),
    #                           client_weights, server_weights)
    #
    #     # Use the client_optimizer to update the local model.
    #     for batch in dataset:
    #         with tf.GradientTape() as tape:
    #             # Compute a forward pass on the batch of data
    #             outputs = model.forward_pass(batch)
    #
    #         # Compute the corresponding gradient
    #         grads = tape.gradient(outputs.loss, client_weights)
    #         grads_and_vars = zip(grads, client_weights)
    #
    #         # Apply the gradient using a client optimizer.
    #         client_optimizer.apply_gradients(grads_and_vars)
    #
    #     return client_weights
    #
    # @tf.function # Vanilla FedAvg
    # def server_update(model, mean_client_weights):
    #     """Updates the server model weights as the average of the client model weights."""
    #     model_weights = model.trainable_variables
    #     # Assign the mean client weights to the server model.
    #     tf.nest.map_structure(lambda x, y: x.assign(y),
    #                           model_weights, mean_client_weights)
    #     return model_weights