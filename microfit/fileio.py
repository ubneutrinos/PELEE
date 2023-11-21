"""Helper function to store and retrieve results from file"""

import numpy as np
import json

# Custom JSON Encoder that can handle numpy arrays and other objects
class MicrofitEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Add a flag to know it was a numpy array
            return {"_obj_type": "ndarray", "data": obj.tolist()}
        else:
            if obj.__class__.__name__ in ["Binning", "Histogram"]:
                return {"_obj_type": obj.__class__.__name__, "data": obj.to_dict()}
        return super(MicrofitEncoder, self).default(obj)


def microfit_decoder(dct):
    if "_obj_type" in dct:
        if dct["_obj_type"] == "ndarray":
            return np.array(dct["data"])
        elif dct["_obj_type"] in ["Binning", "Histogram"]:
            try:
                module = __import__('microfit').histogram
                class_ = getattr(module, dct["_obj_type"])
                return class_.from_dict(dct["data"])
            except ImportError:
                pass
    return dct


def from_json(filename):
    with open(filename, "r") as f:
        return json.load(f, object_hook=microfit_decoder)


def to_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, cls=MicrofitEncoder)
