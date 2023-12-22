import unittest
from ..selections import extract_variables_from_query, find_common_selection

class TestExtractVariablesFromQuery(unittest.TestCase):

    def test_single_variable(self):
        query = "x == 1"
        self.assertEqual(extract_variables_from_query(query), {"x"})
        
    def test_multiple_variables(self):
        query = "x == 1 and y > 2"
        self.assertEqual(extract_variables_from_query(query), {"x", "y"})
        
    def test_repeated_variables(self):
        query = "x == 1 and x > 2"
        self.assertEqual(extract_variables_from_query(query), {"x"})
        
    def test_variables_in_parentheses(self):
        query = "(x > 1) or (y <= 2)"
        self.assertEqual(extract_variables_from_query(query), {"x", "y"})
        
    def test_variables_with_underscore(self):
        query = "var_name == 1"
        self.assertEqual(extract_variables_from_query(query), {"var_name"})
        
    def test_complex_query(self):
        query = "x == 1 and (y > 2 or z < 3) and var_name != 4"
        self.assertEqual(extract_variables_from_query(query), {"x", "y", "z", "var_name"})

class TestFindCommonSelection(unittest.TestCase):
    def test_without_parentheses(self):
        s1 = "x > 0 and y < 5"
        s2 = "x > 0 and y > 5"
        common, (unique1, unique2) = find_common_selection([s1, s2])
        self.assertEqual(common, "x > 0")
        self.assertEqual(unique1, "y < 5")
        self.assertEqual(unique2, "y > 5")

    def test_with_parentheses(self):
        s1 = "x > 0 and y < 5 and (a == 1 or b == 2)"
        s2 = "x > 0 and y > 5 and (a == 1 or b == 2)"
        common, (unique1, unique2) = find_common_selection([s1, s2])
        self.assertEqual(common, " and ".join(sorted(["x > 0", "(a == 1 or b == 2)"])))
        self.assertEqual(unique1, "y < 5")
        self.assertEqual(unique2, "y > 5")

    def test_with_parentheses_and_whitespace(self):
        s1 = "x > 0 and y < 5 and (a == 1 or b == 2)"
        s2 = "x > 0 and y > 5 and (  a == 1 or b == 2   )"
        common, (unique1, unique2) = find_common_selection([s1, s2])
        self.assertEqual(common, " and ".join(sorted(["x > 0", "(a == 1 or b == 2)"])))
        self.assertEqual(unique1, "y < 5")
        self.assertEqual(unique2, "y > 5")
    
    def test_single_condition(self):
        s1 = "x > 0"
        s2 = "x > 0"
        common, (unique1, unique2) = find_common_selection([s1, s2])
        self.assertEqual(common, "x > 0")
        self.assertEqual(unique1, "")
        self.assertEqual(unique2, "")

    def test_nested_parentheses(self):
        s1 = "x > 0 and (y < 5 and (a == 1 or b == 2))"
        s2 = "x > 0 and (y > 5 and (a == 1 or b == 2))"
        common, (unique1, unique2) = find_common_selection([s1, s2])
        self.assertEqual(common, "x > 0")
        self.assertEqual(unique1, "(y < 5 and (a == 1 or b == 2))")
        self.assertEqual(unique2, "(y > 5 and (a == 1 or b == 2))")

    def test_identical_strings(self):
        s1 = "x > 0 and y < 5 and (a == 1 or b == 2)"
        s2 = "x > 0 and y < 5 and (a == 1 or b == 2)"
        common, (unique1, unique2) = find_common_selection([s1, s2])
        self.assertEqual(common, " and ".join(sorted(["x > 0", "y < 5", "(a == 1 or b == 2)"])))
        self.assertEqual(unique1, "")
        self.assertEqual(unique2, "")

    def test_multiple_strings(self):
        s1 = "x > 0 and y < 5 and (a == 1 or b == 2)"
        s2 = "x > 0 and y > 5 and (a == 1 or b == 2)"
        s3 = "x > 0 and z < 3 and (a == 1 or b == 2)"
        common, (unique1, unique2, unique3) = find_common_selection([s1, s2, s3])
        self.assertEqual(common, " and ".join(sorted(["x > 0", "(a == 1 or b == 2)"])))
        self.assertEqual(unique1, "y < 5")
        self.assertEqual(unique2, "y > 5")
        self.assertEqual(unique3, "z < 3")
    
    def test_one_string(self):
        s1 = "x > 0 and y < 5 and (a == 1 or b == 2)"
        common, (unique1,) = find_common_selection([s1])
        self.assertEqual(common, s1)
        self.assertEqual(unique1, "")
    
    def test_empty_list(self):
        common, unique = find_common_selection([])
        self.assertEqual(common, "")
        self.assertEqual(unique, [])
    
    def test_empty_string(self):
        s1 = ""
        s2 = ""
        common, (unique1, unique2) = find_common_selection([s1, s2])
        self.assertEqual(common, "")
        self.assertEqual(unique1, "")
        self.assertEqual(unique2, "")
    
    def test_empty_string_and_nonempty_string(self):
        s1 = ""
        s2 = "x > 0"
        common, (unique1, unique2) = find_common_selection([s1, s2])
        self.assertEqual(common, "")
        self.assertEqual(unique1, "")
        self.assertEqual(unique2, "x > 0")
    
    def test_empty_string_and_multiple_nonempty_strings(self):
        s1 = ""
        s2 = "x > 0"
        s3 = "y < 0"
        s4 = "z == 0"
        common, (unique1, unique2, unique3, unique4) = find_common_selection([s1, s2, s3, s4])
        self.assertEqual(common, "")
        self.assertEqual(unique1, "")
        self.assertEqual(unique2, "x > 0")
        self.assertEqual(unique3, "y < 0")
        self.assertEqual(unique4, "z == 0")
    
    def test_empty_string_and_multiple_nonempty_strings_with_parentheses(self):
        s1 = ""
        s2 = "x > 0 and (y < 0 or z == 0)"
        s3 = "x > 0 and (y < 0 or z == 0)"
        common, (unique1, unique2, unique3) = find_common_selection([s1, s2, s3])
        self.assertEqual(common, "")
        self.assertEqual(unique1, "")
        self.assertEqual(unique2, " and ".join(sorted(["x > 0", "(y < 0 or z == 0)"])))
        self.assertEqual(unique3, " and ".join(sorted(["x > 0", "(y < 0 or z == 0)"])))

if __name__ == "__main__":
    unittest.main()
