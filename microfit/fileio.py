"""Helper function to store and retrieve results from file"""

import numpy as np
import json
from .histogram import Binning, Histogram

# Custom JSON Encoder that can handle numpy arrays and other objects
class MicrofitEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Add a flag to know it was a numpy array
            return {"_obj_type": "ndarray", "data": obj.tolist()}
        elif type(obj) in [Binning, Histogram]:
            return {"_obj_type": type(obj).__name__, "data": obj.to_dict()}
        return super(MicrofitEncoder, self).default(obj)


# Custom JSON Decoder to handle numpy arrays
def microfit_decoder(dct):
    if "_obj_type" in dct:
        if dct["_obj_type"] == "ndarray":
            return np.array(dct["data"])
        elif dct["_obj_type"] == "Binning":
            return Binning.from_dict(dct["data"])
        elif dct["_obj_type"] == "Histogram":
            return Histogram.from_dict(dct["data"])
    return dct


def from_json(filename):
    with open(filename, "r") as f:
        return json.load(f, object_hook=microfit_decoder)


def to_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, cls=MicrofitEncoder)
