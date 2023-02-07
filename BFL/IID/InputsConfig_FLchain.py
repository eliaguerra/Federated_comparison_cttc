import math

import tensorflow_federated as tff
from datetime import datetime
import tensorflow as tf
import os 
import matplotlib.pyplot as plt
import collections
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class InputsConfig:
    """ Seclect the model to be simulated.
    0 : The base model
    1 : Bitcoin model
    2 : Ethereum model
    3 : AppendableBlock model
    4 : FLchain model
    """
    
    model = 4

    ''' Input configurations for the base model '''
    if model == 0:
        ''' Block Parameters '''
        Binterval = 600  # Average time (in seconds)for creating a block in the blockchain
        Bsize = 1.0  # The block size in MB
        Bdelay = 0.42  # average block propogation delay in seconds, #Ref: https://bitslog.wordpress.com/2016/04/28/uncle-mining-an-ethereum-consensus-protocol-flaw/
        Breward = 12.5  # Reward for mining a block

        ''' Transaction Parameters '''
        hasTrans = True  # True/False to enable/disable transactions in the simulator
        Ttechnique = "Light"  # Full/Light to specify the way of modelling transactions
        Tn = 10  # The rate of the number of transactions to be created per second
        # The average transaction propagation delay in seconds (Only if Full technique is used)
        Tdelay = 5.1
        Tfee = 0.000062  # The average transaction fee
        Tsize = 0.000546  # The average transaction size  in MB

        ''' Node Parameters '''
        Nn = 3  # the total number of nodes in the network
        NODES = []
        from Models.Node import Node
        # here as an example we define three nodes by assigning a unique id for each one
        NODES = [Node(id=0), Node(id=1)]

        ''' Simulation Parameters '''
        simTime = 1000  # the simulation length (in seconds)
        Runs = 2  # Number of simulation runs

    ''' Input configurations for Bitcoin model '''
    if model == 1:
        ''' Block Parameters '''
        Binterval = 600  # Average time (in seconds)for creating a block in the blockchain
        Bsize = 1.0  # The block size in MB
        Bdelay = 0.42  # average block propogation delay in seconds, #Ref: https://bitslog.wordpress.com/2016/04/28/uncle-mining-an-ethereum-consensus-protocol-flaw/
        Breward = 12.5  # Reward for mining a block

        ''' Transaction Parameters '''
        hasTrans = True  # True/False to enable/disable transactions in the simulator
        Ttechnique = "Light"  # Full/Light to specify the way of modelling transactions
        Tn = 10  # The rate of the number of transactions to be created per second
        # The average transaction propagation delay in seconds (Only if Full technique is used)
        Tdelay = 5.1
        Tfee = 0.000062  # The average transaction fee
        Tsize = 0.000546  # The average transaction size  in MB

        ''' Node Parameters '''
        Nn = len(emnist_train.client_ids)  # the total number of nodes in the network
        NODES = []
        from Models.Bitcoin.Node import Node
        # here as an example we define three nodes by assigning a unique id for each one + % of hash (computing) power
        NODES = [Node(id=0, hashPower=50), Node(
            id=1, hashPower=20), Node(id=2, hashPower=30)]

        ''' Simulation Parameters '''
        simTime = 10000  # the simulation length (in seconds)
        Runs = 1  # Number of simulation runs

    ''' Input configurations for Ethereum model '''
    if model == 2:
        ''' Block Parameters '''
        Binterval = 12.42  # Average time (in seconds)for creating a block in the blockchain
        Bsize = 1.0  # The block size in MB
        Blimit = 8000000  # The block gas limit
        Bdelay = 6  # average block propogation delay in seconds, #Ref: https://bitslog.wordpress.com/2016/04/28/uncle-mining-an-ethereum-consensus-protocol-flaw/
        Breward = 2  # Reward for mining a block

        ''' Transaction Parameters '''
        hasTrans = True  # True/False to enable/disable transactions in the simulator
        Ttechnique = "Light"  # Full/Light to specify the way of modelling transactions
        Tn = 20  # The rate of the number of transactions to be created per second
        # The average transaction propagation delay in seconds (Only if Full technique is used)
        Tdelay = 3
        # The transaction fee in Ethereum is calculated as: UsedGas X GasPrice
        Tsize = 0.000546  # The average transaction size  in MB

        ''' Drawing the values for gas related attributes (UsedGas and GasPrice, CPUTime) from fitted distributions '''

        ''' Uncles Parameters '''
        hasUncles = True  # boolean variable to indicate use of uncle mechansim or not
        Buncles = 2  # maximum number of uncle blocks allowed per block
        Ugenerations = 7  # the depth in which an uncle can be included in a block
        Ureward = 0
        UIreward = Breward / 32  # Reward for including an uncle

        ''' Node Parameters '''
        Nn = 3  # the total number of nodes in the network
        NODES = []
        from Models.Ethereum.Node import Node
        # here as an example we define three nodes by assigning a unique id for each one + % of hash (computing) power
        NODES = [Node(id=0, hashPower=50), Node(
            id=1, hashPower=20), Node(id=2, hashPower=30)]

        ''' Simulation Parameters '''
        simTime = 500  # the simulation length (in seconds)
        Runs = 2  # Number of simulation runs

    ''' Input configurations for AppendableBlock model '''
    if model == 3:
        ''' Transaction Parameters '''
        hasTrans = True  # True/False to enable/disable transactions in the simulator

        Ttechnique = "Full"

        # The rate of the number of transactions to be created per second
        Tn = 10

        # The maximum number of transactions that can be added into a transaction list
        txListSize = 100

        ''' Node Parameters '''
        # Number of device nodes per gateway in the network
        Dn = 10
        # Number of gateway nodes in the network
        Gn = 2
        # Total number of nodes in the network
        Nn = Gn + (Gn * Dn)
        # A list of all the nodes in the network
        NODES = []
        # A list of all the gateway Ids
        GATEWAYIDS = [chr(x + 97) for x in range(Gn)]
        from Models.AppendableBlock.Node import Node

        # Create all the gateways
        for i in GATEWAYIDS:
            otherGatewayIds = GATEWAYIDS.copy()
            otherGatewayIds.remove(i)
            # Create gateway node
            NODES.append(Node(i, "g", otherGatewayIds))

        # Create the device nodes for each gateway
        deviceNodeId = 1
        for i in GATEWAYIDS:
            for j in range(Dn):
                NODES.append(Node(deviceNodeId, "d", i))
                deviceNodeId += 1

        ''' Simulation Parameters '''
        # The average transaction propagation delay in seconds
        propTxDelay = 0.000690847927

        # The average transaction list propagation delay in seconds
        propTxListDelay = 0.00864894

        # The average transaction insertion delay in seconds
        insertTxDelay = 0.000010367235

        # The simulation length (in seconds)
        simTime = 500

        # Number of simulation runs
        Runs = 5

        ''' Verification '''
        # Varify the model implementation at the end of first run
        VerifyImplemetation = True

        maxTxListSize = 0

    ''' Input configurations for FLchain (PoW) model '''
    if model == 4:

        ''' FL Parameters '''
        FL_TYPE = 1             # '1': Synchronous, '2': Asynchronous
        NUM_CLIENTS = 200       # number of clients to be selected at each iteration (FL_TYPE=1)
        #NUM_CLIENTS_TEST = 1   # number of clients for test evaluation
        TEST_CLIENTS = 3183
        NUM_EPOCHS = 5          # number of local epochs
        BATCH_SIZE = 20         # batch size local datasets
        SHUFFLE_BUFFER = 100    # shuffling
        PREFETCH_BUFFER = 10    # prefetching
        NUM_ROUNDS_FL = 200     # max. number of rounds in FL
        MODEL = "CNN"           # CNN or FFNN
        SEED = 0
        # Learning rates (client/server)
        LEARNING_RATE_CLIENT = 0.02
        LEARNING_RATE_SERVER = 1.00

        # Load the MNIST dataset
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data() #Download the mnist version with only digits 
        emnist_eval, emnist_test = tff.simulation.datasets.ClientData.train_test_client_split(emnist_test, TEST_CLIENTS, seed=0)
                        
        # TODO: Pruning the dataset to generate non-iid-ness
        # pruned_emnist_train = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
        #     client_ids_list_ix, create_tf_dataset_for_client_fn)


        ''' Node Parameters '''
        Nn = len(emnist_train.client_ids)  # the total number of nodes in the network
        Nm = 10  # the total number of miners in the network
        NODES = []
        from Models.Bitcoin.Node import Node
        client_ids_list = emnist_train.client_ids
        # Generate a blockchain client (generating transactions) for each FL client
        for i in range(0, len(client_ids_list)):
            if i < Nm:  # Select the first Nm clients to also act as miners
                NODES.append(Node(id=i, flAddress=client_ids_list[i], hashPower=100 / Nm))
            else:
                NODES.append(Node(id=i, flAddress=client_ids_list[i], hashPower=0))

        ''' Transaction Parameters '''
        hasTrans = True             # True/False to enable/disable transactions in the simulator
        Ttechnique = "Light"        # Full/Light to specify the way of modelling transactions
        capacityP2P = 100/8         # Capacity of P2P links in MBps
        capacityNode = 1/8          # Capacity of client links in MBps
        txListSize = NUM_CLIENTS    # The maximum number of transactions that can be added into a transaction list
        Tsize = 2.33                # transaction size (in MB)
        Tdelay = Tsize/capacityNode # The average transaction propagation delay in s (Only if Full technique is used)
        Tfee = 0.000062             # The average transaction fee

        ''' Block Parameters '''
        MiningRate = 2.5                 # Mining capacity in Hz
        Binterval = 15                   # Average time (in seconds) for creating a block in the blockchain
        Bh = 0.2/8                       # Block header size in MB
        Bsize = (txListSize * Tsize) + 1    # The block size in MB
        Bdelay = (Bh+Bsize)/capacityP2P  # Average block propagation delay in seconds
        Breward = 12.5                   # Reward for mining a block

        # The rate of the number of transactions to be created per second
        avgClientData = 100
        sigma = 0.1e-4
        Tn = txListSize / Binterval 

        ''' Simulation Parameters '''
        simTime = 100000000                 # the simulation length (in seconds)
        Runs = 1                            # Number of simulation runs
        maxBlockSimulation = 450            # maximum number of blocks 
                
        path = f"EMNIST_performance_{MODEL}/NUM_TEST_CLIENTS_{NUM_CLIENTS}_NUM_EPOCHS_{NUM_EPOCHS}_BATCH_SIZE_{BATCH_SIZE}_NUM_ROUNDS_{NUM_ROUNDS_FL}_MINING_RATE_{MiningRate}_BINTERVAL_{Binterval}/"

