"""Module to define parameters of an analysis."""

from dataclasses import dataclass, asdict
from typing import Union, List, Tuple
import unittest  # Enclosed List in quotes
from unitpy import Unit, Quantity
import toml
from numbers import Number
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
    is_discrete: bool = None  # Automatically assigned in constructor
    bounds: Tuple[Quantity] = None
    _post_init_finished: bool = False

    def __post_init__(self):
        if isinstance(self.value, bool):
            self.is_discrete = True
        else:
            self.is_discrete = False
        # convert float to Quantity
        if isinstance(self.value, Number) and not isinstance(self.value, bool):
            self.value = Quantity(self.value, Unit())
        # if bounds are given, convert to Quantity if needed, assuming the same units as value
        if self.bounds is not None:
            if isinstance(self.bounds[0], Number):
                self.bounds = [Quantity(b, self.value.unit) for b in self.bounds]
            # make sure bounds are sorted
            if self.bounds[0] > self.bounds[1]:
                self.bounds = [self.bounds[1], self.bounds[0]]
        self._post_init_finished = True

    @property
    def m(self):
        """Magnitude of the parameter value."""
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
                if isinstance(value, Number) and not isinstance(value, bool):
                    value = Quantity(value, self.value.unit)
                # make sure new value is within bounds
                if self.bounds is not None:
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
        if isinstance(d["value"], dict):
            if "unit" in d["value"]:
                d["value"] = Quantity(d["value"]["magnitude"], Unit(d["value"]["unit"]))
            else:
                d["value"] = Quantity(d["value"]["magnitude"], Unit())
        return cls(**d)
    
    def copy(self):
        return Parameter.from_dict(self.to_dict())


class ParameterSet:
    def __init__(self, parameters: List["Parameter"]):
        """Class to define a set of parameters of an analysis.

        Parameters
        ----------
        parameters : List[Parameter]
            List of parameters.
        """
        self.parameters = parameters
        self._check_duplicates()

    def _check_duplicates(self):
        # If a parameter with the same name is defined twice, it must
        # be defined identically. In that case, we remove the duplicate.
        # If there is a contradiction between the two definitions, we
        # raise an error.

        if len(self.parameters) == len(set(self.names)):
            return

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

    def __add__(self, other):
        return ParameterSet(self.parameters + other.parameters)

    def __radd__(self, other):
        if other == 0:
            return self
        return ParameterSet(self.parameters + other.parameters)

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

    @classmethod
    def from_toml(cls, toml_path):
        if isinstance(toml_path, str):
            with open(toml_path, "r") as f:
                dict_from_toml = toml.load(f)
        else:
            dict_from_toml = toml_path
        return cls.from_dict(dict_from_toml)

    def to_toml(self, toml_path):
        with open(toml_path, "w") as f:
            toml.dump(self.to_dict(), f)

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

class TestParameter(unittest.TestCase):
    def test_discrete_parameter(self):
        p = Parameter(name="discrete", value=True)
        self.assertTrue(p.is_discrete)
        self.assertEqual(p.value, True)

    def test_continuous_parameter(self):
        p = Parameter(name="continuous", value=Quantity(1.0, Unit("m")))
        self.assertFalse(p.is_discrete)
        self.assertEqual(p.value, Quantity(1.0, Unit("m")))

    def test_bounds(self):
        p = Parameter(
            name="bounded", value=Quantity(1.0, Unit("m")), bounds=(Quantity(0.0, Unit("m")), Quantity(2.0, Unit("m")))
        )
        self.assertEqual(p.bounds, (Quantity(0.0, Unit("m")), Quantity(2.0, Unit("m"))))

    def test_magnitude(self):
        p = Parameter(name="magnitude", value=Quantity(1.0, Unit("m")))
        self.assertEqual(p.m, 1.0)
    
    def test_dict_conversion(self):
        p = Parameter(name="float", value=3.0)
        d = p.to_dict()
        p2 = Parameter.from_dict(d)
        self.assertEqual(p, p2)

        p = Parameter(name="float_with_unit", value=Quantity(2.0, Unit("m")))
        d = p.to_dict()
        p2 = Parameter.from_dict(d)
        self.assertEqual(p, p2)

        # test bool parameter
        p = Parameter(name="bool", value=True)
        d = p.to_dict()
        p2 = Parameter.from_dict(d)
        self.assertEqual(p, p2)

    def test_copy(self):
        p = Parameter(name="float", value=3.0)
        p2 = p.copy()
        self.assertEqual(p, p2)
        self.assertIsNot(p, p2)

        p = Parameter(name="float_with_unit", value=Quantity(2.0, Unit("m")))
        p2 = p.copy()
        self.assertEqual(p, p2)
        self.assertIsNot(p, p2)

        # test bool parameter
        p = Parameter(name="bool", value=True)
        p2 = p.copy()
        self.assertEqual(p, p2)
        self.assertIsNot(p, p2)

class TestParameterSet(unittest.TestCase):
    def test_insertion(self):
        p1 = Parameter(name="float", value=3.0)
        p2 = Parameter(name="float_with_unit", value=Quantity(2.0, Unit("m")))
        p3 = Parameter(name="bool", value=True)
        p4 = Parameter(name="float", value=3.0)

        ps = ParameterSet([p1, p2, p3, p4])

        # Test that the set only contains one instance of each parameter
        self.assertEqual(len(ps), 3)

        # Test that an error is raised if a parameter is inserted twice with different values
        with self.assertRaises(ValueError):
            p5 = Parameter(name="float", value=4.0)
            ps = ParameterSet([p1, p2, p3, p4, p5])

    def check_shared_params(self, param_sets: List[ParameterSet], shared_names: List[str]):
        # Reference ParameterSet
        ref_set = param_sets[0]
        for name in shared_names:
            ref_param = ref_set[name]
            for ps in param_sets[1:]:
                assert ref_param is ps[name], f"Parameter {name} is not the same object in all ParameterSets"

    def test_parameter_sharing(self):
        p1 = Parameter("x", Quantity(1.0, Unit("m")), bounds=(0.0, 2.0))
        p2 = Parameter("y", 2.0, bounds=(0.0, 5.0))
        p3 = Parameter("y", 2.0, bounds=(0.0, 5.0))
        p4 = Parameter("w", True)

        ps = ParameterSet([p1, p2, p3])
        ps2 = ParameterSet([p1, p2, p4])

        class ParameterUser:
            def __init__(self, parameters, name=None):
                self.parameters = parameters
                self.name = name

        class ParameterUserUser:
            def __init__(self, parameter_users: List[ParameterUser]):
                self.parameter_users = parameter_users
                self.parameters = sum([pu.parameters for pu in parameter_users])

        pu1 = ParameterUser(ps, name="pu1")
        pu2 = ParameterUser(ps2, name="pu2")
        puu = ParameterUserUser([pu1, pu2])
        puu.parameters["x"] = 0.5
        # Setting the parameter value in the ParameterUserUser should also change the value in
        # each ParameterUser
        for pu in [pu1, pu2]:
            assert pu.parameters["x"].m == 0.5
        puu.parameters["w"] = False
        # Setting the value of a parameter that only exists in one ParameterUser should
        # not change the value in the other ParameterUser or add the parameter to it.
        # Assert that accessing the parameter in the other ParameterUser raises a KeyError.
        with self.assertRaises(KeyError):
            pu1.parameters["w"]
        assert pu2.parameters["w"].value == False, pu2.parameters["w"]

        shared_names = list(set(pu1.parameters.names).intersection(set(pu2.parameters.names)))
        self.check_shared_params([pu1.parameters, pu2.parameters], shared_names)
        self.check_shared_params([pu1.parameters, pu2.parameters, puu.parameters], list(shared_names))
        # When we instantiate a new object and inject it into the parameter set of one of the 
        # parameter users, the objects will no longer be the same and our test should fail.
        pu2.parameters.parameters[0] = Parameter("x", Quantity(0.7, Unit("m")), bounds=(0.0, 2.0))
        with self.assertRaises(AssertionError):
            self.check_shared_params([pu1.parameters, pu2.parameters, puu.parameters], list(shared_names))


    def test_parameter_sharing_multi_level(self):
        p1 = Parameter("x", Quantity(1.0, Unit("m")), bounds=(0.0, 2.0))
        p2 = Parameter("y", 2.0, bounds=(0.0, 5.0))
        p3 = Parameter("z", 3.0, bounds=(0.0, 6.0))

        ps1 = ParameterSet([p1])
        ps2 = ParameterSet([p2])
        ps3 = ParameterSet([p3])

        class ParameterUser:
            def __init__(self, parameters, name=None):
                self.parameters = parameters
                self.name = name

        class ParameterUserUser:
            def __init__(self, parameter_users: List[ParameterUser]):
                self.parameter_users = parameter_users
                self.parameters = sum([pu.parameters for pu in parameter_users])

        pu1 = ParameterUser(ps1, name="pu1")
        pu2 = ParameterUser(ps2, name="pu2")
        pu3 = ParameterUser(ps3, name="pu3")

        puu1 = ParameterUserUser([pu1, pu2])
        puu2 = ParameterUserUser([pu2, pu3])

        puu_top = ParameterUserUser([puu1, puu2])

        # Test that the value change propagates to all nested ParameterUserUser and ParameterUser instances
        puu_top.parameters["x"] = 0.7
        for pu in [pu1, pu2, pu3, puu1, puu2, puu_top]:
            try:
                assert pu.parameters["x"].m == 0.7
            except KeyError:
                pass  # Skip those ParameterUser(s) that don't have "x"

    def test_dict_conversion(self):
        p1 = Parameter(name="float", value=3.0)
        p2 = Parameter(name="float_with_unit", value=Quantity(2.0, Unit("m")))
        p3 = Parameter(name="bool", value=True)

        ps = ParameterSet([p1, p2, p3])

        d = ps.to_dict()
        ps2 = ParameterSet.from_dict(d)
        self.assertEqual(ps, ps2)

    def test_copy(self):
        p1 = Parameter(name="float", value=3.0)
        p2 = Parameter(name="float_with_unit", value=Quantity(2.0, Unit("m")))
        p3 = Parameter(name="bool", value=True)

        ps = ParameterSet([p1, p2, p3])
        ps2 = ps.copy()
        self.assertEqual(ps, ps2)
        self.assertIsNot(ps, ps2)
    
    def test_get_list(self):
        # test indexing with a list of names
        p1 = Parameter(name="float", value=3.0)
        p2 = Parameter(name="float_with_unit", value=Quantity(2.0, Unit("m")))
        p3 = Parameter(name="bool", value=True)
        
        ps = ParameterSet([p1, p2, p3])
        ps2 = ps[["float", "bool"]]
        self.assertEqual(ps2, ParameterSet([p1, p3]))
        # the objects should still point to the same Parameter objects
        self.assertIs(ps2.parameters[0], ps.parameters[0])
    
    def test_empty_equality(self):
        # make sure the empty set is equal to itself
        ps1 = ParameterSet([])
        ps2 = ParameterSet([])
        self.assertEqual(ps1, ps2)

if __name__ == "__main__":
    unittest.main()
