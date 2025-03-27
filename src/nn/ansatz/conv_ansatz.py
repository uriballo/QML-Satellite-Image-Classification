import pennylane as qml
import torch
from pennylane import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_num_params_QCNN_multiclass(wires, params):
    
    n_qubits = len(wires)

    dropped_wires = params.get('dropped_wires',[])
    n_drop_init = len(dropped_wires)
    layers = params.get('layers',np.log2(((n_qubits-len(dropped_wires)))).astype(int))
    layers_FC = params.get('layers_FC',15)
    
    add_fully_connected = params.get('add_fully_connected',True)

    conv_params = 15
    pool_params = 3

    n_params_total = 0
    active_qubits = n_qubits-n_drop_init
    params_per_layer = {}

    for layer in range(1,layers+1):
        layer_qubits = active_qubits + n_drop_init
        index_param_block = {} # dict to store the index where each block starts
        n_params = 0
        conv_blocks = layer_qubits

        n_conv_params = int(conv_blocks*conv_params)
        n_params += n_conv_params # parameters per convolution block, update n_params
        index_param_block[f'conv'] = (n_params_total,n_params_total + n_conv_params) # index where convolution starts and ends
        
        # if odd number of qubits, drop one
        if active_qubits % 2 == 1:
            active_qubits -= 1
            n_drop_init += 1

        pool_blocks = active_qubits//2
        n_pool_params = int(pool_blocks*pool_params)
        n_params += n_pool_params # parameters per pooling block, update n_params
        active_qubits = pool_blocks
        index_param_block[f'pool'] = (n_params + n_params_total - n_pool_params,n_params + n_params_total) # index where pooling starts and ends (after convolution)

        n_params_total += n_params # update n_params_total
        params_per_layer[f'layer_{layer}'] = index_param_block
    
    if add_fully_connected:

        n_qubits_fully_connected = n_drop_init + active_qubits
        params_fully_connected = (n_qubits_fully_connected-1)*layers_FC*2+n_qubits_fully_connected

        params_per_layer['fully_connected'] = (n_params_total,n_params_total+params_fully_connected)
        n_params_total += params_fully_connected

    return n_params_total, params_per_layer


def QCNN_multiclass(wires, params):   
    
    # fetch parameters 
    dropped_wires = params.get('dropped_wires')
    excluded_wires = params.get('excluded_wires',[])
    n_qubits = len(wires)
    if dropped_wires is None:
        dropped_w = []
    else:
        dropped_w = dropped_wires.copy()

    active_wires = [wire for wire in wires if wire not in (dropped_w + excluded_wires)]

    add_fully_connected = params.get('add_fully_connected',True)
    
    # get number of layers   
    layers = params.get('layers',np.log2(((n_qubits-len(dropped_w)))).astype(int))
    layers_FC = params.get('layers_FC',15)

    # get nb parameters
    nb_params_total,params_per_layer = get_num_params_QCNN_multiclass(wires,params)

    weights = params.get('weights',torch.rand(nb_params_total, device = device)) * 2* torch.pi - torch.pi

    for layer in range(1,layers+1):
        
        # #############################
        # #### Convolutional layer ####
        # #############################

        # set wires for convolutional layer
        conv_wires = active_wires + dropped_w
        conv_wires.sort()
        n_conv_wires = len(conv_wires)
    
        # fetch the weights for the convolutional layer
        start_index,end_index = params_per_layer[f'layer_{layer}']['conv']
        weights_conv = weights[start_index:end_index]
        weights_conv = weights_conv.reshape(-1, 4**2-1)
        weight_index = 0      

        for i in range(0, n_conv_wires, 2):
            qml.ArbitraryUnitary(weights_conv[weight_index], wires=[conv_wires[i], conv_wires[(i + 1)% n_conv_wires]])
            weight_index += 1
            qml.Barrier() 
            
        if n_conv_wires !=2:
            for i in range(1, n_conv_wires, 2):
                qml.ArbitraryUnitary(weights_conv[weight_index], wires=[conv_wires[i], conv_wires[(i + 1)% n_conv_wires]])
                weight_index += 1
                qml.Barrier() 

        ##### end of conv layer #####

        # if not even number of wires for pool, drop one
        if len(active_wires) % 2 != 0:
            dropped_w.append((active_wires).pop())

        ############################
        ###### Pooling  layer ######
        ############################

        # define the wires for the pooling layer
        padd = len(active_wires) // 2
        pool_wires_sinks = active_wires[padd:]
        pool_wires_sources = active_wires[:-padd]

        # fetch the weights for the pooling layer
        start_index,end_index = params_per_layer[f'layer_{layer}']['pool']
        weights_pool = weights[start_index:end_index]
        weights_pool = weights_pool.reshape(-1, 3)
        weight_index = 0

        for source, sink in zip(pool_wires_sources, pool_wires_sinks):
            m_outcome = qml.measure(sink)
            qml.cond(m_outcome, qml.U3)(*weights_pool[weight_index], source)
            weight_index += 1
            qml.Barrier()

        # update active wires
        active_wires = list(pool_wires_sources)

        #### end of pool  layer ####

    ############################
    ## Fully connected layer ###
    ############################

    if add_fully_connected:
        start_index,end_index = params_per_layer['fully_connected']
        wires_fully_connected = active_wires + dropped_w
        n_wires_fully_connected = len(wires_fully_connected)
        end_index_init = start_index + n_wires_fully_connected

        weights_init = weights[start_index:end_index_init]
        weights_layers_fully_connected = weights[end_index_init:].reshape(
            layers_FC,n_wires_fully_connected-1,2)

        qml.SimplifiedTwoDesign(initial_layer_weights=weights_init,
                                weights=weights_layers_fully_connected,
                                wires=wires_fully_connected)
        qml.Barrier()
        
    ##### end of  FC layer #####