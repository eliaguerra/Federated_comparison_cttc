from importlib.resources import path
from threading import local
import numpy as np
from InputsConfig_FLchain import InputsConfig as p
from Event import Queue
from Statistics import Statistics
import json
import tensorflow_datasets as tfds
import tensorflow as tf

np.random.seed(0)


if p.model == 4:
    from Models.Bitcoin.BlockCommit import BlockCommit
    from Models.Bitcoin.Consensus import Consensus
    from Models.Transaction import LightTransaction as LT, FullTransaction as FT
    from Models.Bitcoin.Node import Node
    from Models.Incentives import Incentives
    from Models.FederatedLearning import FederatedLearning
    import tensorflow_federated as tff
    import time
    from carbontracker.tracker import CarbonTracker
    import os
    from matplotlib import pyplot as plt 

elif p.model == 3:
    from Models.AppendableBlock.BlockCommit import BlockCommit
    from Models.Consensus import Consensus
    from Models.AppendableBlock.Transaction import FullTransaction as FT
    from Models.AppendableBlock.Node import Node
    from Models.Incentives import Incentives
    from Models.AppendableBlock.Statistics import Statistics

elif p.model == 2:
    from Models.Ethereum.BlockCommit import BlockCommit
    from Models.Ethereum.Consensus import Consensus
    from Models.Ethereum.Transaction import LightTransaction as LT, FullTransaction as FT
    from Models.Ethereum.Node import Node
    from Models.Ethereum.Incentives import Incentives

elif p.model == 1:
    from Models.Bitcoin.BlockCommit import BlockCommit
    from Models.Bitcoin.Consensus import Consensus
    from Models.Transaction import LightTransaction as LT, FullTransaction as FT
    from Models.Bitcoin.Node import Node
    from Models.Incentives import Incentives

elif p.model == 0:
    from Models.BlockCommit import BlockCommit
    from Models.Consensus import Consensus
    from Models.Transaction import LightTransaction as LT, FullTransaction as FT
    from Models.Node import Node
    from Models.Incentives import Incentives


########################################################## Start Simulation ##############################################################


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for i in range(p.Runs):
        clock = 0  # set clock to 0 at the start of the simulation
        if p.hasTrans:
            if p.Ttechnique == "Light":
                LT.create_transactions()  # generate pending transactions
            elif p.Ttechnique == "Full":
                FT.create_transactions()  # generate pending transactions

        Node.generate_gensis_block()  # generate the gensis block for all miners
        # initiate initial events >= 1 to start with
        BlockCommit.generate_initial_events()

        #i = 0
        t = 0
        while not Queue.isEmpty() and clock <= p.simTime and Statistics.totalBlocks < p.maxBlockSimulation:
            #print(f"Iteration {i}")
            #i = i + 1
            next_event = Queue.get_next_event()
            clock = next_event.time  # move clock to the time of the event
            BlockCommit.handle_event(next_event)
            # ######################################### #
            # # - - - - - - - - - - - - - - - - - - - # #
            # #    [[FLchain - Added 10-02-2022]]     # #
            # # - - - - - - - - - - - - - - - - - - - # #
            # -----> ASYNCHRONOUS FL OPERATION (TO BED ONE)
            # if p.model == 4 and p.FL_TYPE == 2 and next_event.type == "receive_block":
            #     # 1 - Extract local updates from the latest block
            #     # ... TODO
            #     block_weights = []
            #     # 2 - Aggregate the updates directly at the node
            #     mean_block_weights = tff.federated_mean(block_weights)
            #     aggregate_weights = tff.federated_map(FL.server_update_fn, mean_block_weights)
            #     # 3 - Each client computes their updated weights
            #     client_weights = tff.federated_map(
            #         FL.client_update_fn, (p.emnist_train, aggregate_weights))
            # ######################################### #
            if next_event.type == "create_block":
                print(f"Created block {t} / Total blocks: {Statistics.totalBlocks}")
                t = t+1
            Queue.remove_event(next_event)

        Consensus.fork_resolution()  # apply the longest chain to resolve the forks
        # distribute the rewards between the participating nodes
        Incentives.distribute_rewards()
        # calculate the simulation results (e.g., block statistics and miners' rewards)
        Statistics.calculate()
        
        print(f"Main chian blocks: {Statistics.mainBlocks}")
        # ######################################### #
        # # - - - - - - - - - - - - - - - - - - - # #
        # #    [[FLchain - Added 10-02-2022]]     # #
        # # - - - - - - - - - - - - - - - - - - - # #
        # -----> SYNCHRONOUS FL OPERATION
        if p.model == 4 and p.FL_TYPE == 1:
            # Initialize the FL model
            FL = FederatedLearning()
            iterative_process, state, fed_evaluation, fed_evaluation_data = FL.initialize_fl_model()
            # Iterate for each mined block
            
            training_accuracy_log = []
            training_loss_log = []
            validation_accuracy_log = []
            validation_loss_log = []
            round_time_log = []
            local_time_log = []
            round_time_ml_log = []
            tracker = CarbonTracker(epochs = p.NUM_ROUNDS_FL, epochs_before_pred=-1, monitor_epochs=p.NUM_ROUNDS_FL, log_dir=p.path, devices_by_pid = True, update_interval = 2)
            round_counter = 0
            previous_block_timestamp = 0
            for b_idx, b in enumerate(Consensus.global_chain):
                print("Block #" + str(b.depth) + ": " + str(len(b.transactions)) + " tr.")
                if len(b.transactions) > 0 and round_counter < p.NUM_ROUNDS_FL: # added stopping condition according to the number of FL rounds
                    tracker.epoch_start()
                    round_time_start = time.time()
                    state, train_metrics, val_metrics, local_time = \
                        FL.execute_fl_round(iterative_process, state, fed_evaluation, fed_evaluation_data, b.transactions)
                    
                    # Save performance to statistics
                    #Statistics.train_accuracy.append(train_metrics['train']['sparse_categorical_accuracy'])
                    #Statistics.test_accuracy.append(test_metrics['sparse_categorical_accuracy'])
                    #Statistics.eval_accuracy.append(eval_metrics['sparse_categorical_accuracy'])

                    #log model performance 
                    training_accuracy = train_metrics["train"]["sparse_categorical_accuracy"]
                    training_loss = train_metrics["train"]["loss"]
                    validation_accuracy = val_metrics["sparse_categorical_accuracy"]
                    validation_loss = val_metrics["loss"]
                    
                    training_accuracy_log.append(training_accuracy)
                    training_loss_log.append(training_loss)
                    validation_accuracy_log.append(validation_accuracy)
                    validation_loss_log.append(validation_loss)
                    
                    local_time_log.append(local_time)
                    round_time_end = time.time()
                    blockchain_delay = b.timestamp - Consensus.global_chain[b_idx-1].timestamp
                    round_time = round_time_end - round_time_start + blockchain_delay
                    round_time_ml = round_time_end - round_time_start
                    round_time_ml_log.append(round_time_ml)
                    round_time_log.append(round_time)
                    print(f"Round {round_counter}, training accuracy: {training_accuracy}, validation accuracy: {validation_accuracy}, training time: {local_time}, iteration time: {round_time}")
                    #print(round_time_ml)
                    tracker.epoch_end()
                    round_counter = round_counter + 1

            tracker.stop()
            os.makedirs(p.path, exist_ok=True)
            #save results, compute total duration and plots
            plt.figure()
            plt.title("Accuracy")
            plt.plot(range(0, len(training_accuracy_log)), training_accuracy_log, label = "Training accuracy")
            plt.plot(range(0, len(validation_accuracy_log)), validation_accuracy_log, label="Validation accuracy")
            plt.legend()
            plt.grid()
            plt.savefig(p.path + "performance.jpg")

            plt.figure()
            plt.title("Execution time")
            plt.plot(range(0,len(round_time_log)), round_time_log, label = "Round Time")
            plt.plot(range(0,len(local_time_log)), local_time_log, label = "Local iteration time")
            plt.legend()
            plt.grid()
            plt.savefig(p.path + "time.jpg")
            
            
            #Save the model 
            keras_model = FL.create_keras_model()
            state.model.assign_weights_to(keras_model)
            keras_model.save(p.path + "model.h5")
            keras_model = tf.keras.models.load_model(p.path + "model.h5")
            keras_model.compile(
                optimizer = tf.keras.optimizers.SGD(learning_rate = 0.02),
                loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
            )

            #validation set check
            x_eval = []
            y_eval = []
            emnist_eval_all_clients = p.emnist_eval.create_tf_dataset_from_all_clients(seed=0)
            emnist_eval_all_clients = tfds.as_numpy(emnist_eval_all_clients)
            for sample in emnist_eval_all_clients:
                if p.MODEL == "FFNN":
                    x_eval.append(sample["pixels"].reshape(784))
                    y_eval.append(sample["label"].reshape(1))
                elif p.MODEL == "CNN":
                    x_eval.append(sample["pixels"].reshape((28,28,1)))
                    y_eval.append(sample["label"].reshape(1))

            x_eval = np.array(x_eval)
            y_eval = np.array(y_eval)

            keras_eval_dict = keras_model.evaluate(x=x_eval, y=y_eval, return_dict = True, verbose = 0)
            print(f"Validation accuracy keras model: {keras_eval_dict['sparse_categorical_accuracy']}")

            #test set usage 
            x_test = []
            y_test = []

            emnist_test_all_clients = p.emnist_test.create_tf_dataset_from_all_clients(seed=0)
            emnist_test_all_clients = tfds.as_numpy(emnist_test_all_clients)
            for sample in emnist_test_all_clients:
                if p.MODEL == "FFNN":
                    x_test.append(sample["pixels"].reshape(784))
                    y_test.append(sample["label"].reshape(1))
                elif p.MODEL == "CNN":
                    x_test.append(sample["pixels"].reshape((28,28,1)))
                    y_test.append(sample["label"].reshape(1))
            
            x_test = np.array(x_test)
            y_test = np.array(y_test)
            test_performance = keras_model.evaluate(x=x_test, y=y_test, return_dict = True, verbose = 0)
            print(f"Test set accuracy: {test_performance['sparse_categorical_accuracy']}")


            np.save(p.path + "training_accuracy", np.array(training_accuracy_log))
            np.save(p.path + "training_loss", np.array(training_accuracy_log))
            np.save(p.path + "validation_accuracy", np.array(validation_accuracy_log))
            np.save(p.path + "validation_loss", np.array(validation_loss_log))
            np.save(p.path + "local_time", np.array(local_time_log))
            np.save(p.path + "round_time", np.array(round_time_log))
            np.save(p.path + "round_time_ml", np.array(round_time_ml_log))
            np.save(p.path + "test_loss", np.array(test_performance["loss"]))
            np.save(p.path + "test_accuracy", np.array(test_performance["sparse_categorical_accuracy"]))

            params_dict = {
                "FL_TYPE": p.FL_TYPE,
                "NUM_CLIENTS": p.NUM_CLIENTS,
                "TEST_CLIENTS": p.TEST_CLIENTS,
                "NUM_EPOCHS": p.NUM_EPOCHS,
                "BATCH_SIZE": p.BATCH_SIZE,
                "SHUFFLE_BUFFER": p.SHUFFLE_BUFFER,
                "PREFETCH_BUFFER": p.PREFETCH_BUFFER,
                "NUM_ROUNDS_FL": p.NUM_ROUNDS_FL,
                "LEARNING_RATE_CLIENT": p.LEARNING_RATE_CLIENT,
                "LEARNING_RATE_SERVER": p.LEARNING_RATE_SERVER,
                "Nn": p.Nn,
                "Nm":p.Nm,
                "hasTrans": p.hasTrans,
                "Ttechnique": p.Ttechnique,
                "capacityP2P": p.capacityP2P,
                "capacityNode": p.capacityNode,
                "txListSize": p.txListSize,
                "Tsize": p.Tsize,
                "Tdelay": p.Tdelay,
                "Tfee": p.Tfee,
                "MiningRate": p.MiningRate,
                "Binterval": p.Binterval,
                "Bh": p.Bh,
                "Bsize": p.Bsize,
                "Bdelay": p.Bdelay,
                "Breward": p.Breward,
                "avgClientData": p.avgClientData,
                "sigma": p.sigma,
                "Tn": p.Tn,
                "simTime": p.simTime,
                "Runs": p.Runs,
                "maxBlockSimulation": p.maxBlockSimulation
            }
            with open(p.path + "parameters.json", "w") as write_file:
                json.dump(params_dict, write_file, indent=4)

        # ######################################### #

        if p.model == 3:
            Statistics.print_to_excel(i, True)
            Statistics.reset()
        else:
            ########## reset all global variable before the next run #############
            Statistics.reset()  # reset all variables used to calculate the results
            Node.resetState()  # reset all the states (blockchains) for all nodes in the network
            fname = p.path + "(Allverify)1day_{0}M_{1}K.xlsx".format(
                p.Bsize / 1000000, p.Tn / 1000)
            # print all the simulation results in an excel file
            Statistics.print_to_excel(fname)
            fname = "(Allverify)1day_{0}M_{1}K.xlsx".format(
                p.Bsize / 1000000, p.Tn / 1000)
            # print all the simulation results in an excel file
            Statistics.print_to_excel(p.path + fname)
            Statistics.reset2()  # reset profit results


######################################################## Run Main method #####################################################################
if __name__ == '__main__':
    main()
