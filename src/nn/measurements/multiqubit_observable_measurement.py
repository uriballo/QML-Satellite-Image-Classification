import pennylane as qml
from pennylane.pauli import PauliWord,string_to_pauli_word
import itertools
from pennylane import numpy as np

def get_pauli_multiqubit_observables(wires,params):
    meas_wires = params.get('meas_wires',wires)
    n_obs = params.get('n_obs',0)
    pauli_op_string = params.get('pauli_op_string','Z')
    observables = [{} for _ in range(n_obs)]
    count = 0
    for i in meas_wires:
        for obs_i in range(n_obs):
            if count < (obs_i+1)*(len(meas_wires))/n_obs:
                observables[obs_i][i] = pauli_op_string
                
                count += 1
                break
    return [PauliWord(obs).operation() for obs in observables]

def get_pauli_words(wires,params):
    '''Function to get the Pauli words for n qubits.

    Args:
        - wires (int) : wires onto which the pauli words will be defined
        - params (dict) : remaining args
    Returns:
    list: list of Pauli words for n qubits'''
    n_wires = len(wires)
    wire_map = {p:i for i,p in enumerate(wires)}
    pauli_strs = ['I', 'X', 'Y', 'Z']
    pauli_words = [''.join(p) for p in itertools.product(pauli_strs, repeat=n_wires)]
    pauli_words.remove(n_wires*'I')
    pauli_words = [string_to_pauli_word(p,wire_map) for p in pauli_words]
    return pauli_words

def random_pauli_string_over_meas_wires(wires,params):
    meas_wires = params.get('meas_wires',wires)
    n_obs = params.get('n_obs',0)
    pauli_words = get_pauli_words(meas_wires,params)
    seed = params.get('seed',42)
    # shuffle the pauli words
    np.random.seed(seed)
    np.random.shuffle(pauli_words)

    return pauli_words[:n_obs]
    

def measurement_multiqubit(wires,params):
    observables = params.get('observables',[])
    return [qml.expval(obs) for obs in observables]