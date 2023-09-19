"""Helper function to store and retrieve results from file"""

import numpy as np
import json


# Custom JSON Encoder that can handle numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Add a flag to know it was a numpy array
            return {"_is_numpy_array": True, "data": obj.tolist()}
        return super(NumpyEncoder, self).default(obj)


# Custom JSON Decoder to handle numpy arrays
def numpy_decoder(dct):
    if "_is_numpy_array" in dct:
        return np.array(dct["data"])
    return dct


def from_json(filename):
    with open(filename, "r") as f:
        return json.load(f, object_hook=numpy_decoder)


def to_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)
