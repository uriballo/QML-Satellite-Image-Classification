import pennylane as qml

def probs_measurement(wires,params):
    meas_wires = params.get('meas_wires',wires)

    return qml.probs(wires=meas_wires)