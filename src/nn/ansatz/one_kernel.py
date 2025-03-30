import torch
import pennylane as qml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_num_params_one_kernel(wires,params):
    n_qubits = len(wires)
    conv_params = params.get('conv_params',15) # This is for the 2-qubit arbitrary unitary
    pool_params = params.get('pool_params',3)
    layers = params.get('layers',n_qubits-1)
    layers_FC = params.get('layers_FC',15)
    add_fully_connected = params.get('add_fully_connected',True)
    
    n_params_total = layers*(conv_params+pool_params)
    params_per_layer = {
        f'layer_{layer}': {
            'conv': (0+(layer-1)*(conv_params+pool_params),
                     conv_params+(layer-1)*(conv_params+pool_params)),
            'pool': (conv_params+(layer-1)*(conv_params+pool_params),
                     conv_params+pool_params+(layer-1)*(conv_params+pool_params))}
        for layer in range(1,layers+1)}
    
    if add_fully_connected:
        n_qubits_fully_connected = n_qubits-layers
        params_fully_connected = (n_qubits_fully_connected-1)*layers_FC*2+n_qubits_fully_connected

        params_per_layer['fully_connected'] = (n_params_total,
                                               n_params_total+params_fully_connected)
        n_params_total += params_fully_connected

    return n_params_total,params_per_layer


def one_kernel(wires,params):
    # fetch params
    n_qubits = len(wires)
    layers = params.get('layers',n_qubits-1)
    layers_FC = params.get('layers_FC',15)
    add_fully_connected = params.get('add_fully_connected',True)
    
    # get nb parameters
    nb_params_total,params_per_layer = get_num_params_one_kernel(wires,params)

    weights = params.get('weights',torch.rand(nb_params_total, device = device)) * 2* torch.pi - torch.pi
    
    traced_out_qubits = []
    qubit_i0 = 0
    qubit_i1 = 1
    for layer in range(1,layers+1):
        ### Convolutional layer
        start_index,end_index = params_per_layer[f'layer_{layer}']['conv']
        weights_conv = weights[start_index:end_index]
        
        qml.ArbitraryUnitary(weights_conv, wires=[wires[qubit_i0],wires[qubit_i1]])

        ### Pooling layer
        start_index,end_index = params_per_layer[f'layer_{layer}']['pool']
        weights_pool = weights[start_index:end_index]
        
        m_outcome = qml.measure(wires[qubit_i0])
        qml.cond(m_outcome, qml.U3)(*weights_pool,wires[qubit_i1])

        qubit_i0 +=1
        qubit_i1 += 1
    
    if add_fully_connected:
        start_index,end_index = params_per_layer['fully_connected']
        wires_fully_connected = wires[layers:]
        n_wires_fully_connected = len(wires_fully_connected)
        end_index_init = start_index + n_wires_fully_connected
        
        weights_init = weights[start_index:end_index_init]
        weights_layers_fully_connected = weights[end_index_init:].reshape(
            layers_FC,n_wires_fully_connected-1,2)

        qml.SimplifiedTwoDesign(initial_layer_weights=weights_init,
                                weights=weights_layers_fully_connected,
                                wires=wires_fully_connected)