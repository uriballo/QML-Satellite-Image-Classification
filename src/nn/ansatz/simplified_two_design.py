import pennylane as qml
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_num_params_simplified_two_design(wires,params):
    layers = params.get('layers',15)
    n_qubits = len(wires)

    # get nb parameters
    nb_params_total = (n_qubits-1)*layers*2+n_qubits
    return nb_params_total,{}


def simplified_two_design(wires,params):
    # fetch params
    layers = params.get('layers',15)

    # get nb parameters
    nb_params_total,_ = get_num_params_simplified_two_design(wires,params)

    # fetch weights
    weights = params.get('weights',torch.rand(nb_params_total, device = device)) * 2* torch.pi - torch.pi
    weights_init = weights[:len(wires)]
    weights = weights[len(wires):].reshape(layers,len(wires)-1,2)
    
    qml.SimplifiedTwoDesign(initial_layer_weights=weights_init,weights=weights,wires=wires)