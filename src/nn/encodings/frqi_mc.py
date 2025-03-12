import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import Operation, AnyWires

# decimal to binary list with leading zeros
def dec2bin(n, num_bits=4):
    return [int(x) for x in bin(n)[2:].zfill(num_bits)]

def R_R(theta_R,wires):
    assert len(wires) == 3
    wires = [wires[1],wires[2],wires[0]]
    op = qml.prod(qml.CRY(theta_R,wires=[wires[0],wires[2]]),qml.CNOT(wires=[wires[0],wires[1]]))
    op = qml.prod(op,qml.CRY(-theta_R,wires=[wires[1],wires[2]]))
    op = qml.prod(op,qml.CNOT(wires=[wires[0],wires[1]]))
    op = qml.prod(op,qml.CRY(theta_R,wires=[wires[1],wires[2]]))
    return op

def R_G(theta_G,wires):
    assert len(wires) == 3
    wires = [wires[1],wires[2],wires[0]]
    op = qml.prod(qml.X(wires=wires[1]),qml.CRY(theta_G,wires=[wires[0],wires[2]]))
    op = qml.prod(op,qml.CNOT(wires=[wires[0],wires[1]]))
    op = qml.prod(op,qml.CRY(-theta_G,wires=[wires[1],wires[2]]))
    op = qml.prod(op,qml.CNOT(wires=[wires[0],wires[1]]))
    op = qml.prod(op,qml.CRY(theta_G,wires=[wires[1],wires[2]]))
    op = qml.prod(op,qml.X(wires=wires[1]))
    return op

def R_B(theta_B,wires):
    assert len(wires) == 3
    wires = [wires[1],wires[2],wires[0]]
    op = qml.prod(qml.X(wires=wires[0]),qml.CRY(theta_B,wires=[wires[0],wires[2]]))
    op = qml.prod(op,qml.CNOT(wires=[wires[0],wires[1]]))
    op = qml.prod(op,qml.CRY(-theta_B,wires=[wires[1],wires[2]]))
    op = qml.prod(op,qml.CNOT(wires=[wires[0],wires[1]]))
    op = qml.prod(op,qml.CRY(theta_B,wires=[wires[1],wires[2]]))
    op = qml.prod(op,qml.X(wires=wires[0]))
    return op

def R_prime(theta_R,theta_G,theta_B,wires):
    # op = [R_B(theta_B,wires)]
    # op.append(R_G(theta_G,wires))
    # op.append(R_R(theta_R,wires))
    R_B(theta_B,wires)
    R_G(theta_G,wires)
    R_R(theta_R,wires)

def R_i_mat(theta_R,theta_G,theta_B,i,wires):
    ''' Control R_primer on the ith state of the 2n-dim hilbert space'''
    n = len(wires)-3
    # f = qml.ctrl(R_prime,control=wires[3:],control_values=dec2bin(i,n))
    # f(theta_R,theta_G,theta_B,wires[:3])
    # return f
    op = qml.prod(R_prime)
    return qml.ops.op_math.Controlled(op(theta_R,theta_G,theta_B,wires[:3]),
                                      control_wires=wires[3:],control_values=dec2bin(i,n))

class FRQI_MC(Operation):
    num_params = 1
    num_wires = AnyWires
    grad_method = None

    def __init__(self, image, wires=None,params={}, id=None):
        shape = qml.math.shape(image)[-1:]
        n_features = shape[0]

        img_pixels = params.get('img_pixels', 8)
         
        self.image = image
        self.img_pixels = img_pixels
        self._hyperparameters = {'img_pixels': img_pixels}
        super().__init__(image, wires=wires, id=id)

    
    @staticmethod
    def compute_decomposition(features, wires,img_pixels):

        # add batch dimension if not present
        if qml.math.ndim(features) == 3:
            features = qml.math.expand_dims(features, axis=0)

        # add channel dimension if not present
        if qml.math.ndim(features) == 2:
            features = features.reshape((features.shape[0], 3, img_pixels, img_pixels))

        # initial hadamard operations
        ops = []
        for k in range(len(wires)-1):
            ops.append(qml.Hadamard(wires=wires[k+1]))

        i = 0
        for j in range(img_pixels):
            for k in range(img_pixels):
                ops.append(R_i_mat(features[:,0,j,k],features[:,1,j,k],features[:,2,j,k],i,wires))
                i += 1
        return ops
    
    # def expand(self):
    #     with qml.tape.QuantumTape() as tape:
    #         for feature in self.image:
    #             FRQI_MC.compute_decomposition(feature, wires=self.wires,img_pixels=self.img_pixels)
    #     return tape