import pennylane as qml
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_num_params_two_kernels(wires,params):
    n_qubits = len(wires)
    conv_params = params.get('conv_params',15) # This is for the 2-qubit arbitrary unitary
    pool_params = params.get('pool_params',3)
    layers_default = int((n_qubits-4)/2+1) # We need at least 4 active qubits at each layer, and we kill 2 at each layer
    layers = params.get('layers',layers_default)
    layers_FC = params.get('layers_FC',15)
    add_fully_connected = params.get('add_fully_connected',True)
    
    n_params_total = 2*layers*(conv_params+pool_params)
    params_per_layer = {}
    for op in range(2):
        increment = op * layers*(conv_params+pool_params)
        for layer in range(1,layers+1):
            params_per_layer[f'layer_{layer}_op_{op}'] = {
                'conv': (increment + (layer-1)*(conv_params+pool_params),
                         increment + conv_params+(layer-1)*(conv_params+pool_params)),
                'pool': (increment + conv_params+(layer-1)*(conv_params+pool_params),
                         increment + conv_params+pool_params+(layer-1)*(conv_params+pool_params))
            }

    if add_fully_connected:
        n_qubits_fully_connected = n_qubits-2*layers
        params_fully_connected = (n_qubits_fully_connected-1)*layers_FC*2+n_qubits_fully_connected

        params_per_layer['fully_connected'] = (n_params_total,
                                               n_params_total+params_fully_connected)
        n_params_total += params_fully_connected

    
    return n_params_total,params_per_layer


def two_kernels(wires,params):
    n_qubits = len(wires)
    assert n_qubits >= 4, f'Number of qubits is less than 4: {n_qubits}'

    layers_default = int((n_qubits-4)/2+1) # We need at least 4 active qubits at each layer, and we kill 2 at each layer
    layers = params.get('layers',layers_default)
    layers_FC = params.get('layers_FC',15)
    add_fully_connected = params.get('add_fully_connected',True)
    
    # get nb parameters
    nb_params_total,params_per_layer = get_num_params_two_kernels(wires,params)

    weights = params.get('weights',torch.rand(nb_params_total, device = device)) * 2* torch.pi - torch.pi

    traced_out_qubits = []
    for layer in range(1,layers+1):
        active_wires = [i for i in wires if i not in traced_out_qubits]

        for op_i,op in enumerate(['op_0','op_1']):
            start_index,end_index = params_per_layer[f'layer_{layer}_'+op]['conv']
            weights_conv_op_i = weights[start_index:end_index]
        
            qml.ArbitraryUnitary(weights_conv_op_i, wires=active_wires[op_i*2:op_i*2+2])

        pool_wires_sources = [active_wires[0],active_wires[2]]
        pool_wires_sinks = [active_wires[1],active_wires[3]]
        
        for op_i,op in enumerate(['op_0','op_1']):
            start_index,end_index = params_per_layer[f'layer_{layer}_'+op]['pool']
            weights_pool_op_i = weights[start_index:end_index]

            m_outcome = qml.measure(pool_wires_sources[op_i])
            qml.cond(m_outcome, qml.U3)(*weights_pool_op_i,pool_wires_sinks[op_i])
        
        traced_out_qubits += pool_wires_sources
    
    if add_fully_connected:
        start_index,end_index = params_per_layer['fully_connected']
        wires_fully_connected = [i for i in wires if i not in traced_out_qubits]
        n_wires_fully_connected = len(wires_fully_connected)
        end_index_init = start_index + n_wires_fully_connected
        
        weights_init = weights[start_index:end_index_init]
        weights_layers_fully_connected = weights[end_index_init:].reshape(
            layers_FC,n_wires_fully_connected-1,2)

        qml.SimplifiedTwoDesign(initial_layer_weights=weights_init,
                                weights=weights_layers_fully_connected,
                                wires=wires_fully_connected)