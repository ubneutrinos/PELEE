import unittest
from ..selections import extract_variables_from_query

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

if __name__ == "__main__":
    unittest.main()
