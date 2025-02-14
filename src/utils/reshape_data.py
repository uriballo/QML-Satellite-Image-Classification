import torch
import numpy as np

def reshape_data_flatten(x,n_wires):
    return x.reshape((x.shape[0],-1))

def get_mapping_random(img_pixels,seed=41):
    mapping_img_to_psi = {}
    mapping_psi_to_img = {}
    order = np.array([i for i in range(img_pixels*img_pixels)])
    np.random.seed(seed)
    np.random.shuffle(order)
    k = 0
    for v_pos in range(img_pixels):
        for h_pos in range(img_pixels):
            mapping_psi_to_img[(v_pos,h_pos)] = order[k]
            mapping_img_to_psi[order[k]] = (v_pos,h_pos)
            k += 1
    return mapping_psi_to_img,mapping_img_to_psi

def get_mapping_flat(img_pixels):
    mapping_img_to_psi = {}
    mapping_psi_to_img = {}
    order = np.array([i for i in range(img_pixels*img_pixels)])
    k = 0
    for v_pos in range(img_pixels):
        for h_pos in range(img_pixels):
            mapping_psi_to_img[(v_pos,h_pos)] = k
            mapping_img_to_psi[k] = (v_pos,h_pos)
            k += 1
    return mapping_psi_to_img,mapping_img_to_psi

def get_mapping_vertical_and_horizontal(img_pixels):
    mapping_img_to_psi = {}
    mapping_psi_to_img = {}
    # k = 0
    count = set()
    for x in range(img_pixels):
        for y in range(img_pixels):
                mapping_psi_to_img[(x,y)] = np.floor(y/4).astype(int)*4*img_pixels+y%4+x*4
                mapping_img_to_psi[np.floor(y/4).astype(int)*4*img_pixels+y%4+x*4] = (x,y)
    return mapping_psi_to_img,mapping_img_to_psi

def get_mapping_square_and_vertical(img_pixels):
    mapping_img_to_psi = {}
    mapping_psi_to_img = {}
    k = 0
    v_outside_range = img_pixels//4
    h_outside_range = img_pixels//2
    for i_v_outside in range(v_outside_range):
        for i_h in range(h_outside_range):
            for i_v_inside in range(4):
                for i_h_inside in range(2):
                    mapping_psi_to_img[(i_v_outside*4+i_v_inside,i_h*2+i_h_inside)] = k
                    mapping_img_to_psi[k] = (i_v_outside*4+i_v_inside,i_h*2+i_h_inside)
                    k += 1
    return mapping_psi_to_img,mapping_img_to_psi

class ReshapeDATA:
    '''
    Reshapes data according to 4 different geometric shapes that implement different kernels over the images. It can act on data with 4 dimensions: (batch_index,channels,img_pos_x,img_pos_y).
    '''
    def __init__(self,wires,params):
        self.structure = params.get('structure','flat')
        if self.structure == 'flat':
            self.reshape = self._flatten
        else:
            self.img_pixels = params.get('img_pixels',None)
            self.n_wires = len(wires)
            self.mapping,_ = self._get_mapping()
            self.reshape = self._reshape

    def _get_mapping(self,):
        if self.structure == 'random':
            mapping_psi_to_img,mapping_img_to_psi = get_mapping_random(self.img_pixels)
        elif self.structure == 'vertical_and_horizontal':
            mapping_psi_to_img,mapping_img_to_psi = get_mapping_vertical_and_horizontal(self.img_pixels)
        elif self.structure == 'square_and_vertical':
            mapping_psi_to_img,mapping_img_to_psi = get_mapping_square_and_vertical(self.img_pixels)
        elif self.structure == 'flat':
            mapping_psi_to_img,mapping_img_to_psi = {},{}
        else:
            raise ValueError(f'Mapping structure {self.structure} not implemented. Choose from ["random","vertical_and_horizontal","square_and_vertical","flat".')
        
        return mapping_psi_to_img,mapping_img_to_psi

    def _flatten(self,x):
        return x.reshape((x.shape[0],-1))
        
    def _reshape(self,x):
        batch_size,n_channels, img_size,img_size = x.shape
        new_image = torch.zeros((batch_size,2**self.n_wires))
        padd = img_size*img_size
        for channel in range(n_channels):
            for v_pos in range(img_size):
                for h_pos in range(img_size):
                    new_image[:,self.mapping[h_pos,v_pos]+padd*channel] = x[:,channel,v_pos,h_pos]
        
        return new_image
