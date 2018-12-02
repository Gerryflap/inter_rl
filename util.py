import numpy as np
import base64


def np_to_dict(array):
    d = {
        "size": array.shape,
        "array": base64.encodebytes(array.astype(dtype=np.float32).tobytes()).decode('UTF-8')
    }
    return d


def weights_to_base64(weights):
    weight_strings = []
    for weight_m in weights:
        d = np_to_dict(weight_m)
        weight_strings.append(d)
    return weight_strings


def np_from_dict(array_dict):
    weights_size = array_dict["size"]
    weight_string = array_dict["array"]
    weights = np.frombuffer(base64.decodebytes(weight_string.encode('UTF-8')), dtype=np.float32)
    return np.resize(weights, weights_size)


def base64_to_weights(weight_strings):
    weights = []
    for weight_dict in weight_strings:
        weights.append(np_from_dict(weight_dict))
    return weights


if __name__ == "__main__":
    import json
    weights = [np.random.normal(0, 1, (3, 5)), np.random.normal(0, 1, (5, 3))]
    print(weights)
    w_s = weights_to_base64(weights)
    js = json.dumps(w_s)
    print(js)
    w_s = json.loads(js)
    print(base64_to_weights(w_s))
