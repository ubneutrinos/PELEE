import unittest

from typing import List, Sequence, Union
from unitpy import Unit, Quantity

from ..parameters import Parameter, ParameterSet


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
            name="bounded",
            value=Quantity(1.0, Unit("m")),
            bounds=(Quantity(0.0, Unit("m")), Quantity(2.0, Unit("m"))),
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

    def check_shared_params(self, param_sets: List[ParameterSet]):
        shared_names = list(
            set(param_sets[0].names).intersection(*[set(ps.names) for ps in param_sets[1:]])
        )
        # Reference ParameterSet
        ref_set = param_sets[0]
        for name in shared_names:
            ref_param = ref_set[name]
            for ps in param_sets[1:]:
                assert (
                    ref_param is ps[name]
                ), f"Parameter {name} is not the same object in all ParameterSets"

    def test_parameter_sharing(self):
        p1 = Parameter("x", Quantity(1.0, Unit("m")), bounds=(0.0, 2.0))
        p2 = Parameter("y", 2.0, bounds=(0.0, 5.0))
        p3 = Parameter("y", 2.0, bounds=(0.0, 5.0))
        p4 = Parameter("w", True)

        ps = ParameterSet([p1, p2, p3])
        ps2 = ParameterSet([p1, p2, p4])

        class ParameterUser:
            def __init__(self, parameters, name=None):
                self.parameters = parameters.copy()
                self.name = name

        class ParameterUserUser:
            def __init__(self, parameter_users: List[ParameterUser]):
                self.parameter_users = parameter_users
                self.parameters = sum([pu.parameters for pu in parameter_users], ParameterSet([]))

        pu1 = ParameterUser(ps, name="pu1")
        pu2 = ParameterUser(ps2, name="pu2")
        puu = ParameterUserUser([pu1, pu2])
        puu.parameters["x"] = 0.5
        # Setting the parameter value in the ParameterUserUser should also change the value in
        # each ParameterUser
        self.check_shared_params([pu1.parameters, pu2.parameters, puu.parameters])
        for pu in [pu1, pu2]:
            assert pu.parameters["x"].m == 0.5
        puu.parameters["w"] = False
        # Setting the value of a parameter that only exists in one ParameterUser should
        # not change the value in the other ParameterUser or add the parameter to it.
        # Assert that accessing the parameter in the other ParameterUser raises a KeyError.
        with self.assertRaises(KeyError):
            pu1.parameters["w"]
        assert pu2.parameters["w"].value == False, pu2.parameters["w"]

        self.check_shared_params([pu1.parameters, pu2.parameters])
        self.check_shared_params([pu1.parameters, pu2.parameters, puu.parameters])
        # When we instantiate a new object and inject it into the parameter set of one of the
        # parameter users, the objects will no longer be the same and our test should fail.
        pu2.parameters.parameters[0] = Parameter("x", Quantity(0.7, Unit("m")), bounds=(0.0, 2.0))
        with self.assertRaises(AssertionError):
            self.check_shared_params([pu1.parameters, pu2.parameters, puu.parameters])

    def test_parameter_sharing_multi_level(self):
        p1 = Parameter("x", Quantity(1.0, Unit("m")), bounds=(0.0, 2.0))
        p2 = Parameter("y", 2.0, bounds=(0.0, 5.0))
        p3 = Parameter("z", 3.0, bounds=(0.0, 6.0))

        ps1 = ParameterSet([p1])
        ps2 = ParameterSet([p2])
        ps3 = ParameterSet([p3])

        class ParameterUser:
            def __init__(self, parameters, name=None, copypars=True):
                self.parameters = parameters.copy() if copypars else parameters
                self.name = name

        class ParameterUserUser:
            def __init__(
                self, parameter_users: Sequence[Union[ParameterUser, "ParameterUserUser"]]
            ):
                self.parameters = sum([pu.parameters for pu in parameter_users], ParameterSet([]))

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

        class ParameterPartialUser:
            def __init__(self, parameters: ParameterSet, forward_parameters: List[str]):
                self.parameters = parameters
                self.pu = ParameterUser(parameters[forward_parameters], copypars=False)

        ppu = ParameterPartialUser(puu_top.parameters, ["x", "y"])
        # the puu should now have all parameters, but its internal ParameterUser should only have "x" and "y"
        assert len(ppu.parameters) == 3
        assert len(ppu.pu.parameters) == 2
        # changing the parameter at the top level should propagate to the internal ParameterUser
        ppu.parameters["x"] = 0.5
        assert ppu.pu.parameters["x"].m == 0.5
        # the forwarded parameters should be the same object
        assert ppu.parameters["x"] is ppu.pu.parameters["x"]

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
