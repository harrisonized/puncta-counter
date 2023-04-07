from os.path import dirname
import re


# Functions
# # dirname_n_times
# # title_to-snake_case
# # camel_to_snake_case
# # flatten_columns
# # json_to_dataframe
# # collapse_dataframe
# # expand_dataframe


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


def json_to_dataframe(json, colnames=[],):
    """
    Example:
    smears = {
        "sSP71_18_SAT_CRE_003.png": [4, 6],
        "sSP71_18_SAT_CRE_006.png": [8, 11],
        "sSP83_24_SAT_CRE_002.png": [3, 4, 5, 6, 7, 9, 12, 13, 14, 15, 16, 18, 19, 20, 22, 24, 25],
        "sSP83_24_SAT_CRE_003.png": [6, 9, 10, 12, 13, 16, 20, 22, 24, 27, 28, 29]
    }
    
    dataframe_from_json(smears, colnames=['filename', 'image_number'])
    +---------------------------+----------------------+ 
    |  filename                 | nuclei_object_number |
    +---------------------------+----------------------+
    | sSP71_18_SAT_CRE_003.png  |   4                  |
    | sSP71_18_SAT_CRE_003.png  |   61                 |
    | sSP71_18_SAT_CRE_006.png  |   8                  |
    | sSP71_18_SAT_CRE_006.png  |   11                 |
    | ...                       |   ...                |
    +---------------------------+----------------------+
    
    """
    df = pd.DataFrame.from_dict(json.items())
    df = df.explode(1)
    
    if len(colnames)>0:
        df.rename(columns=dict(zip(list(df.columns), colnames)), inplace=True)
    
    return df


def collapse_dataframe(df, index_cols, value_cols):
    """
    """
    collapsed = (df
        .groupby(index_cols)[value_cols]
        .agg(list)
        .reset_index()
    )
    return collapsed


def expand_dataframe(df, value_cols):
    """Use this to explode multiple columns if pandas version < 1.3.0.
    This is the opposite operation of collapse_dataframe
    """
    df['tmp'] = df[value_cols].apply(lambda x: list(zip(*x)), axis=1)
    df = df.explode('tmp')
    for idx, col in enumerate(value_cols):
        df[col] = df['tmp'].apply(lambda x: x[idx])
    
    return df.drop(columns=['tmp']).reset_index(drop=True)
