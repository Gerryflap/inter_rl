import numpy as np
import base64
import json
import requests


def np_to_dict(array):
    """
    Converts a numpy array to a dict that can be converted to json easily
    :param array: The numpy array
    :return: A dict containing the array in the custom dict format
    """
    d = {
        "size": array.shape,
        "array": base64.encodebytes(array.astype(dtype=np.float32).tobytes()).decode('UTF-8')
    }
    return d


def weights_to_base64(weights):
    """
    Converts a Keras array of weights to an array in the custom dict layout
    :param weights:
    :return:
    """
    weight_strings = []
    for weight_m in weights:
        d = np_to_dict(weight_m)
        weight_strings.append(d)
    return weight_strings


def np_from_dict(array_dict):
    """
    Creates a numpy array from a custom dict type
    :param array_dict:
    :return:
    """
    weights_size = array_dict["size"]
    weight_string = array_dict["array"]
    weights = np.frombuffer(base64.decodebytes(weight_string.encode('UTF-8')), dtype=np.float32)
    return np.resize(weights, weights_size)


def base64_to_weights(weight_dicts):
    """
    Converts a custom dict array to an array of weights for a Keras model
    :param weight_dicts: The array of weight dicts
    :return: A weight array
    """
    weights = []
    for weight_dict in weight_dicts:
        weights.append(np_from_dict(weight_dict))
    return weights


def report_experience(experiences, server_addr=('localhost', 1337)):
    """
    Reports the (s, a, r, sp, term) tuples to the server
    :param experiences: A list of experience tuples
    :param server_addr: the server address: a tuple of ip and port
    """
    experiences = [
            (
                np_to_dict(s),
                a,
                r,
                np_to_dict(sp),
                term
            )
            for s, a, r, sp, term in experiences
        ]
    exp_json = json.dumps(experiences)
    r = requests.post("http://%s:%d/experience" % server_addr, data={"experiences": exp_json})
    print(r.status_code, r.reason)


def get_model(ks, server_addr=('localhost', 1337)):
    """
    Gets the model parameters from the server and instantiates the model with those parameters
    :param ks: Keras. The imported Keras library. Keras can't be imported here because that will start a TF session
    :param server_addr: the server address: a tuple of ip and port
    :return: The Keras model
    """
    r = requests.get("http://%s:%d/model" % server_addr)
    m_params = r.json()
    return model_from_dict(m_params, ks)


def model_from_dict(m_params, ks):
    """
    Loads a model from the custom dict notation.
    :param m_params: Model dict in custom format
    :param ks: Keras (again, due to import problems)
    :return: The Keras model
    """
    model = ks.models.model_from_json(json.dumps(m_params['layout']))
    model.set_weights(base64_to_weights(m_params['weights']))
    return model

