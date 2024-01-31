"""Module to define parameters of an analysis."""

from dataclasses import dataclass, asdict
from numbers import Real
from typing import Optional, Union, List, Tuple, overload
from unitpy import Unit, Quantity
import numpy as np


@dataclass
class Parameter:
    """Class to define a parameter of an analysis.

    Attributes
    ----------
    name : str
        Name of the parameter.
    value : Union[bool, Quantity]
        Value of the parameter. If the parameter is discrete, the value should be a boolean.
        If the parameter is continuous, the value should be a Quantity.
    is_discrete : bool, optional
        Whether the parameter is discrete or continuous. If not given, it is automatically
        assigned based on the type of value.
    bounds : Tuple[Quantity], optional
        Bounds of the parameter. If not given, the parameter is assumed to be unconstrained.
    """

    name: str
    value: Union[bool, Quantity]
    is_discrete: bool = False  # Automatically assigned in constructor
    bounds: Optional[Tuple[Quantity, Quantity]] = None
    _post_init_finished: bool = False

    def __post_init__(self):
        if isinstance(self.value, bool):
            self.is_discrete = True
        else:
            self.is_discrete = False
        # convert float to Quantity
        if isinstance(self.value, Real) and not isinstance(self.value, bool):
            assert isinstance(self.value, float) or isinstance(self.value, int)
            self.value = Quantity(self.value, Unit())
        # if bounds are given, convert to Quantity if needed, assuming the same units as value
        if self.bounds is not None:
            if isinstance(self.value, bool):
                raise TypeError("Cannot assign bounds to a discrete parameter.")
            if isinstance(self.bounds[0], float) or isinstance(self.bounds[0], int):
                assert isinstance(self.bounds[1], float) or isinstance(self.bounds[1], int)
                self.bounds = (
                    Quantity(self.bounds[0], self.value.unit),
                    Quantity(self.bounds[1], self.value.unit),
                )
            # make sure bounds are sorted
            if self.bounds[0] > self.bounds[1]:
                self.bounds = (self.bounds[1], self.bounds[0])
        self._post_init_finished = True

    @property
    def magnitude_bounds(self):
        if self.bounds is None:
            return None
        return (self.bounds[0].value, self.bounds[1].value)

    @property
    def m(self):
        """Magnitude of the parameter value."""
        if self.is_discrete:
            return self.value
        else:
            assert isinstance(self.value, Quantity)
            return self.value.value

    def to_dict(self):
        d = asdict(self)
        # Iterate through dictionary and handle Quantity
        for key, value in d.items():
            if isinstance(value, Quantity):
                d[key] = {"magnitude": value.value, "unit": str(value.unit)}
        return d

    def __setattr__(self, name, value):
        if name == "value":
            if self._post_init_finished:  # Ensure post_init has run
                if self.is_discrete and not isinstance(value, bool):
                    raise TypeError("Cannot assign non-boolean value to a discrete parameter.")
                elif not self.is_discrete and isinstance(value, bool):
                    raise TypeError("Cannot assign boolean value to a continuous parameter.")
                # if generic number, convert to Quantity, assuming the same units as value
                if isinstance(value, Real) and not isinstance(value, bool):
                    assert isinstance(value, float) or isinstance(value, int)
                    assert isinstance(self.value, Quantity)
                    value = Quantity(value, self.value.unit)
                assert isinstance(value, Quantity) or isinstance(value, bool)
                # make sure new value is within bounds
                if self.bounds is not None and not self.is_discrete:
                    assert isinstance(self.value, Quantity)
                    assert isinstance(value, Quantity)
                    if value < self.bounds[0] or value > self.bounds[1]:
                        raise ValueError(
                            f"New value is not within {self.bounds[0].value, self.bounds[1].value} {self.value.unit}."
                        )
        super().__setattr__(name, value)

    @classmethod
    def from_dict(cls, d):
        if "_post_init_finished" in d:
            d.pop("_post_init_finished")
        # Check if the value is a string with units
        if isinstance(d["value"], str):
            magnitude_str, unit_str = d["value"].split()
            d["value"] = Quantity(float(magnitude_str), Unit(unit_str))
        elif isinstance(d["value"], dict):
            if "unit" in d["value"]:
                d["value"] = Quantity(d["value"]["magnitude"], Unit(d["value"]["unit"]))
            else:
                d["value"] = Quantity(d["value"]["magnitude"], Unit())
        return cls(**d)

    def copy(self):
        return Parameter.from_dict(self.to_dict())


class ParameterSet:
    def __init__(self, parameters: List["Parameter"], strict_duplicate_checking=False):
        """Class to define a set of parameters of an analysis.

        Parameters
        ----------
        parameters : List[Parameter]
            List of parameters.
        """
        self.parameters = parameters
        self._check_duplicates(strict=strict_duplicate_checking)

    def _check_duplicates(self, strict=True):
        # If a parameter with the same name is defined twice, it must
        # be defined identically. In that case, we remove the duplicate.
        # If there is a contradiction between the two definitions, we
        # raise an error.

        if len(self.parameters) == len(set(self.names)):
            return
        elif strict:
            raise ValueError("Duplicate parameter names found (strict checking enabled).")
        names, counts = np.unique(self.names, return_counts=True)
        for name, count in zip(names, counts):
            if count > 1:
                # Find the indices of the parameters with this name
                indices = [i for i, x in enumerate(self.names) if x == name]
                # Check if the parameters are identical
                if not all(self.parameters[i] == self.parameters[indices[0]] for i in indices):
                    raise ValueError(f"Parameter {name} is defined twice with different values.")
                # Remove the duplicates
                for i in indices[1:]:
                    self.parameters.pop(i)

    def __add__(self, other: "ParameterSet"):
        assert isinstance(other, ParameterSet)
        parameter_list = self.parameters + other.parameters
        new_parameter_set = ParameterSet(parameter_list, strict_duplicate_checking=False)
        # before we return, we set the parameter objects in the other set to the
        # parameter objects in the new set
        new_parameter_set.synchronize(other)
        return new_parameter_set

    def synchronize(self, other):
        duplicate_names = list(set(self.names).intersection(other.names))
        for name in duplicate_names:
            other._replace_param(name, self[name])

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def _replace_param(self, key, value):
        """Like setitem, but replaces the actual parameter object instead of setting the value."""
        if isinstance(key, str):
            if key not in self.names:
                raise KeyError(f"Parameter {key} not found.")
            self.parameters[self.names.index(key)] = value
        elif isinstance(key, int):
            self.parameters[key] = value
        else:
            raise TypeError("Invalid argument type.")

    @property
    def values(self):
        return [p.value for p in self.parameters]

    @property
    def magnitudes(self):
        return [p.m for p in self.parameters]

    @property
    def magnitude_bounds(self):
        return [p.magnitude_bounds for p in self.parameters]

    @property
    def is_empty(self):
        return len(self.parameters) == 0

    @property
    def names(self):
        return [p.name for p in self.parameters]

    def to_dict(self):
        return [p.to_dict() for p in self.parameters]

    @classmethod
    def from_dict(cls, d):
        return cls([Parameter.from_dict(p) for p in d])

    # Overloads like these don't actually do anything, but they are useful for type checking.
    # In this way, the type checker knows how the output type depends on the input type.
    @overload
    def __getitem__(self, key: str) -> Parameter:
        ...

    @overload
    def __getitem__(self, key: int) -> Parameter:
        ...

    @overload
    def __getitem__(self, key: List[str]) -> 'ParameterSet':
        ...

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self.names:
                raise KeyError(f"Parameter {key} not found.")
            return self.parameters[self.names.index(key)]
        elif isinstance(key, int):
            return self.parameters[key]
        elif isinstance(key, list):
            return ParameterSet([self.parameters[self.names.index(k)] for k in key])
        else:
            raise TypeError("Invalid argument type.")

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.parameters[self.names.index(key)].value = value
        elif isinstance(key, int):
            self.parameters[key].value = value
        else:
            raise TypeError("Invalid argument type.")

    # the repr should list the parameters as a table of name, value, and unit
    def __repr__(self):
        if self.is_empty:
            return "Parameters: None"
        table_header = "Parameters:\n"
        table_header += "Name\tValue\tUnit\tBounds\n"
        table_header += "-------------------------------\n"
        table_body = ""
        for param in self.parameters:
            if param.is_discrete:
                print_value = param.value
                print_unit = ""
                print_bounds = ""
            else:
                assert isinstance(param.value, Quantity)
                print_value = param.value.value
                print_unit = param.value.unit
                if param.bounds is not None:
                    print_bounds = f"({param.bounds[0].value}, {param.bounds[1].value})"
                else:
                    print_bounds = ""
            table_body += f"{param.name}\t{print_value}\t{print_unit}\t{print_bounds}\n"
        return table_header + table_body

    def __len__(self):
        return len(self.parameters)

    def __eq__(self, other):
        if not isinstance(other, ParameterSet):
            return False
        if len(self) != len(other):
            return False
        if len(self) == 0:  # special case: the empty set is equal to itself
            return True
        # parameters might be in different order between the two sets
        # but that should not be a problem
        for name in self.names:
            if self[name] != other[name]:
                return False
        return True

    def copy(self):
        return ParameterSet([p.copy() for p in self.parameters])
