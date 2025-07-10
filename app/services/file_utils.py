import pandas as pd
import os

def mean(df, column):
    return df[column].mean()

def sum_(df, column):
    return df[column].sum()

def min_(df, column):
    return df[column].min()

def max_(df, column):
    return df[column].max()

def count(df):
    return len(df)

def unique(df, column):
    return df[column].unique().tolist()

def median(df, column):
    return df[column].median()

def std(df, column):
    return df[column].std()

def describe(df, column):
    return df[column].describe().to_dict()

def value_counts(df, column):
    return df[column].value_counts().to_dict()

def filter_rows(df, column, value):
    return df[df[column] == value].to_dict(orient='records')

def groupby_agg(df, group_col, agg_col, agg_func):
    return df.groupby(group_col)[agg_col].agg(agg_func).to_dict()

def sort(df, column, ascending=True):
    return df.sort_values(by=column, ascending=ascending).to_dict(orient='records')

def correlation(df, col1, col2):
    return df[col1].corr(df[col2])

def load_structured_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.csv':
        return pd.read_csv(filename)
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(filename)
    else:
        raise ValueError("Unsupported file type")

def answer_structured_query(df, query):
    # Legacy simple matcher (kept for fallback)
    query = query.lower()
    for col in df.columns:
        if col.lower() in query:
            if "average" in query or "mean" in query:
                return f"Average of {col}: {df[col].mean()}"
            if "sum" in query:
                return f"Sum of {col}: {df[col].sum()}"
            if "max" in query:
                return f"Max of {col}: {df[col].max()}"
            if "min" in query:
                return f"Min of {col}: {df[col].min()}"
            if "unique" in query:
                return f"Unique values in {col}: {df[col].unique()}"
    if "count" in query or "number of rows" in query:
        return f"Number of rows: {len(df)}"
    return None

def toolbox_dispatch(df, function_name, args):
    """
    Call the toolbox function by name with args dict.
    """
    functions = {
        'mean': mean,
        'sum': sum_,
        'min': min_,
        'max': max_,
        'count': count,
        'unique': unique,
        'median': median,
        'std': std,
        'describe': describe,
        'value_counts': value_counts,
        'filter_rows': filter_rows,
        'groupby_agg': groupby_agg,
        'sort': sort,
        'correlation': correlation,
    }
    if function_name not in functions:
        raise ValueError(f"Function '{function_name}' not found in toolbox.")
    return functions[function_name](df, **args) 