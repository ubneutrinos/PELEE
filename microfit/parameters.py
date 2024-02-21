"""Module to define parameters of an analysis."""

from numbers import Real
from typing import Optional, Union, List, Tuple, overload
from unitpy import Unit, Quantity
import numpy as np


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

    def __init__(
        self,
        name: str,
        value: Union[bool, Quantity, int, float],
        bounds: Optional[
            Union[Tuple[Quantity, Quantity], Tuple[float, float], Tuple[int, int]]
        ] = None,
    ):
        self.name = name
        self.value = value
        if isinstance(self.value, bool):
            self.is_discrete = True
        else:
            self.is_discrete = False
        self.bounds = bounds

    @property
    def value(self) -> Union[bool, Quantity]:
        return self._value

    @value.setter
    def value(self, value: Union[bool, Quantity, int, float]):
        if isinstance(value, bool):
            self._value = value
        elif isinstance(value, Quantity):
            self._value = value
        else:
            assert isinstance(value, float) or isinstance(value, int)
            self._value = Quantity(value, Unit())

    @property
    def bounds(self) -> Optional[Tuple[Quantity, Quantity]]:
        return self._bounds

    @bounds.setter
    def bounds(
        self,
        bounds: Optional[Union[Tuple[Quantity, Quantity], Tuple[float, float], Tuple[int, int]]],
    ):
        if bounds is not None and self.is_discrete:
            raise ValueError("Bounds cannot be set for discrete parameters.")
        if bounds is None:
            self._bounds = None  # type: ignore
        elif isinstance(bounds[0], Quantity) and isinstance(bounds[1], Quantity):
            # make sure bounds are sorted
            if bounds[0] > bounds[1]:
                self._bounds: Tuple[Quantity, Quantity] = (bounds[1], bounds[0])
            else:
                self._bounds: Tuple[Quantity, Quantity] = bounds  # type: ignore
        else:
            assert isinstance(bounds[0], float) or isinstance(bounds[0], int)
            assert isinstance(bounds[1], float) or isinstance(bounds[1], int)
            assert isinstance(self.value, Quantity), "Value must be a Quantity if bounds are given."
            self._bounds = (
                Quantity(bounds[0], self.value.unit),
                Quantity(bounds[1], self.value.unit),
            )
            # make sure bounds are sorted
            if self._bounds[0] > self._bounds[1]:
                self._bounds = (self._bounds[1], self._bounds[0])

    @property
    def magnitude_bounds(self):
        if self.bounds is None:
            return None
        return (self.bounds[0].value, self.bounds[1].value)

    @property
    def m(self) -> Union[bool, int, float]:
        """Magnitude of the parameter value."""
        if self.is_discrete:
            assert isinstance(self.value, bool)
            return self.value
        else:
            assert isinstance(self.value, Quantity)
            return self.value.value

    def to_dict(self):
        d = {
            "name": self.name,
            "value": self.value,
            "bounds": self.bounds,
        }
        # Iterate through dictionary and handle Quantity
        for key, value in d.items():
            if isinstance(value, Quantity):
                d[key] = {"magnitude": value.value, "unit": str(value.unit)}
        return d

    @classmethod
    def from_dict(cls, d):
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

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Parameter):
            return False
        return (
            self.name == __value.name
            and self.value == __value.value
            and self.bounds == __value.bounds
        )


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
    def __getitem__(self, key: List[str]) -> "ParameterSet":
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
