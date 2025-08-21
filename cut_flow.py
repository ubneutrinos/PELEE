import re
import textwrap ##For cut flow module
from typing import List, Tuple
import microfit.selections as sel

##Cut flow module developed by Amir Gruber

def do_cut_flow(rundata, preselection, selection, *attributes, printed=False, weighted=True):
    """
    Perform a cut flow analysis on the given rundata for multiple attributes.

    Parameters:
    - rundata (dict): A dictionary containing dataframes.
    - preselection (str): String representing preselection name.
    - selection (str): String representing selection name.
    - printed (bool): Whether to print the cut flow table.
    - attributes (str): Variable length argument list for attribute names.

    Returns:
    - cut_events (dict): Dictionary mapping cut conditions to dictionaries with 'included' and 'excluded' DataFrames.
    """
    # Get preselection and selection cuts
    preselection_cuts, selection_cuts = _get_cuts(preselection, selection)

    # Initial weighted counts before any cuts
    initial_counts = {}
    for attr in attributes:
        if attr not in rundata:
            raise ValueError(f"Attribute '{attr}' not found in rundata")
        if not hasattr(rundata[attr], 'query'):
            raise ValueError(f"rundata['{attr}'] is not a DataFrame or does not support the .query method")
        if weighted:
            if 'weights' not in rundata[attr].columns:
                raise ValueError(f"rundata['{attr}'] does not contain a 'weights' column")
            initial_counts[attr] = rundata[attr]['weights'].sum()
        else:
            initial_counts[attr] = len(rundata[attr])

    def print_cut_flow(step, cut, counts):
        if not printed:
            return
        wrapped_cut = textwrap.wrap(cut, width=30)
        for j, line in enumerate(wrapped_cut):
            if j == 0:
                row = f"{step:<10}{line:<30}"
                for count, initial_count in counts:
                    percentage = (count / initial_count) * 100
                    row += f"{count:<10.2f}{percentage:>6.2f}% "
                print(row)
            else:
                row = f"{'':<10}{line:<30}"
                print(row)

    # Print table header
    if printed:
        header = f"{'Step':<10}{'Cut Condition':<30}"
        init_counts_header = f"{'Cut.0':<10}{'No Cuts':<30}"
        for attr in attributes:
            header += f"{attr:<10}{'Effic.':<10}"
            init_counts_header += f"{initial_counts[attr]:<10.2f}{'100.00%':<10}"
        print(header)
        print("-" * (10 + 30 + 20 * len(attributes)))
        print(init_counts_header)
        print("-" * (10 + 30 + 20 * len(attributes)))

    # Initialize dictionary to store included and excluded events
    cut_events = {attr: {} for attr in attributes}

    # Apply preselection cuts
    selected_data = {attr: rundata[attr].copy() for attr in attributes}
    for i, cut in enumerate(preselection_cuts):
        counts = []
        for attr in attributes:
            prev_data = selected_data[attr]
            selected_data[attr] = selected_data[attr].query(cut)
            excluded_data = prev_data[~prev_data.index.isin(selected_data[attr].index)]
            cut_events[attr][cut] = {
                'included': selected_data[attr],
                'excluded': excluded_data
            }
            if weighted:
                count = selected_data[attr]['weights'].sum()
            else:
                count = len(selected_data[attr])
            counts.append((count, initial_counts[attr]))
        print_cut_flow(f"Pre.{i + 1}", cut, counts)

    if printed:
        print("-" * (10 + 30 + 20 * len(attributes)))

    # Apply selection cuts cumulatively
    for i, cut in enumerate(selection_cuts):
        counts = []
        for attr in attributes:
            prev_data = selected_data[attr]
            selected_data[attr] = selected_data[attr].query(cut)
            excluded_data = prev_data[~prev_data.index.isin(selected_data[attr].index)]
            cut_events[attr][cut] = {
                'included': selected_data[attr],
                'excluded': excluded_data
            }
            if weighted:
                count = selected_data[attr]['weights'].sum()
            else:
                count = len(selected_data[attr])
            counts.append((count, initial_counts[attr]))
        print_cut_flow(f"Sel.{i + 1}", cut, counts)

    return cut_events
        
def _get_cuts(preselection_name, selection_name):
    # Input: names of preselection and selection
    # Output: Two lists of strings, 1 for preselection 2 for selection, each list contains one string for each clause
    
    # Check if selection_name exists in globals()
    if selection_name in sel.selection_categories:
        selection = _split_to_cuts(sel.selection_categories[selection_name]['query'])
    else:
        raise ValueError(f"Selection with name '{selection_name}' not found")
    
    # Check if preselection_name exists in preselection_categories
    if preselection_name in sel.preselection_categories:
        preselection = _split_to_cuts(sel.preselection_categories[preselection_name]['query'])
    else:
        raise ValueError(f"Preselection with name '{preselection_name}' not found")
    
    # Returns only the unique cuts in selection
    return _compare_lists(preselection, selection)

def _split_to_cuts(expression):
    # Define the regular expression pattern to split on 'and' outside parentheses
    pattern = r'\s+and\s+(?![^(]*\))'
    # Split the expression based on the pattern
    clauses = re.split(pattern, expression)
    # Remove leading and trailing whitespace from each clause and filter out empty strings
    clauses = [clause.strip() for clause in clauses if clause.strip()]
    return clauses

def _compare_lists(preselection, selection):
    # Check if every item in preselection is also in selection
    for item in preselection:
        if item not in selection:
            raise ValueError(f"Cut '{item}' found in preselection but not in selection")

    # Find unique cuts in selection
    unique_in_selection = [item for item in selection if item not in preselection]

    return preselection, unique_in_selection