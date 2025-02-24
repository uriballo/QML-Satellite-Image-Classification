from src.nn.encodings.IQP_embedding import custom_iqp_embedding
from src.nn.encodings.NQE_embedding import NQE_embedding
from src.nn.encodings.ring_embedding import ring_embedding
from src.nn.encodings.waterfall_embedding import waterfall_embedding
from src.nn.encodings.pennylane_templates import amplitude_embedding, angle_embedding, QAOA_embedding

def load_params(embedding_type, circuit, measurement, limit, dataset, qkernel_shape, use_quantum, image_size):
    if embedding_type == "angle":
        # Angle embedding
        embedding = angle_embedding
        params = {}

    elif embedding_type == "amplitude":
        # Amplitude embedding
        embedding = amplitude_embedding
        params = {}

    elif embedding_type == "IQP":
        # IQP embedding
        embedding = custom_iqp_embedding
        params = {"embedding": {"n_repeats": 2}}

    elif embedding_type == "NQE":
        # NQE embedding
        embedding = NQE_embedding
        params = {"embedding": {"n_repeats": 2}}

    elif embedding_type == "QAOA":
        # Squeezing embedding
        embedding = QAOA_embedding
        params = {"embedding":{"qkernel_shape": 2}}

    elif embedding_type == "ring":
        # Ring embedding
        embedding = ring_embedding
        params = {"embedding":{"n_repeats": 1}}

    elif embedding_type == "waterfall":
        # Ring embedding
        embedding = waterfall_embedding
        params = {"embedding":{"n_repeats": 1}}

    if use_quantum:
        prename = "quantum_" + dataset + "_"
        name_run =  embedding_type + f" {circuit.__name__} limit = {limit}"
        quanv_params = {
        "embedding": embedding,
        "circuit": circuit,
        "measurement": measurement,
        "params": params,
        }
        
    else:
        prename = "classic_"+ dataset + "_"
        name_run = f"classic limit = {limit}"
        quanv_params = None

    if (dataset == "DeepSat4" or dataset == "DeepSat6"):
        if use_quantum:
            in_channels_1, in_channels_2, kernel_size_1, kernel_size_2 = 3, 16, 7, 27
            if image_size == 16:
                kernel_size_2 = 21
            if qkernel_shape == 3:
                in_channels_2, kernel_size_2 = 36, 26
        else:
            in_channels_1, in_channels_2, kernel_size_1, kernel_size_2 = 4, 32, 28, 27
            if image_size == 16:
                kernel_size_2 = 21
    elif dataset == "EuroSAT":
        in_channels_1, in_channels_2, kernel_size_1, kernel_size_2 = 3, 3 * qkernel_shape**2, 7, 7

    name_prueba = dataset + " " + str(image_size) + "x" + str(image_size)
    return in_channels_1, in_channels_2, qkernel_shape, kernel_size_1, kernel_size_2, quanv_params,  name_run, name_prueba, prename
