from os.path import dirname
import re


# Functions
# # dirname_n_times
# # title_to-snake_case
# # camel_to_snake_case
# # flatten_columns



def dirname_n_times(path, n=1):
    for i in range(n):
        path = dirname(path)
    return path


def title_to_snake_case(text):
    """Converts "Column Title" to column_title
    """
    return text.lower().replace(' ', '_').replace('-', '_')


def camel_to_snake_case(text):
    """
    | Converts columnTitle to column_title
    | Source: Geeks For Geeks
    """
    split_First = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    all_lower_case = re.sub('([a-z0-9])([A-Z])', r'\1_\2', split_First).lower()
    return all_lower_case


def flatten_columns(multicols):
    """Flattens a 2 level multi-index
    """
    return [f'{cols[0].lower()}_{cols[1]}'.strip('_') for cols in multicols]