import pennylane as qml

def default_measurement(wires, params):
    observable = params.get('observable', qml.PauliZ)
    return [qml.expval(observable(w)) for w in wires]
