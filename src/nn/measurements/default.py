import pennylane as qml

def default_measurement(wires, params):
    observable = params.get('observable', qml.PauliZ)
    meas_wires = params.get('meas_wires',wires)
    return [qml.expval(observable(w)) for w in meas_wires]