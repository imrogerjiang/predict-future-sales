import pandas as pd
import numpy as np

def downcast_df(df, categorical_cols=(), verbose="none"):
    """
    parameters
    verbose = None, "dataframe", "columns"
    """
    
    
    """
    Casts dataframe column to most compact types.

    Args:
        df: Dataframe to be compressed.
        categorical_cols: Iterable of categorical column names.
        na_col_suffix: Suffix of bool columns indicating nans from converted integer columns
        verbose: How much information to output


    Returns: 
        out: input dataframe appeneded with vector for bag of words
        bow_reference: dataframe with bag of words index
    """
#   If max_value exceeds half of integer range, use type one larger
    int_ranges = {
        (0     , 2**7) :np.uint8,
        (-2**6 , 2**6) :np.int8,
        (0     , 2**15):np.uint16,
        (-2**14, 2**14):np.int16,
        (0     , 2**31):np.uint32,
        (-2**30, 2**30):np.int32,
        (0     , 2**63):np.uint64,
        (-2**62, 2**62):np.int64,
    }
    
    float_precisions = {
        (-2**10, 2**10):np.float16,
        (-2**23, 2**23):np.float32,
        (-2**52, 2**52):np.float64,
    }
        
        
    na_columns = []
    
    start_size = df.memory_usage(deep=True).sum()//(1024)**2

    for c in df.columns:
        if verbose=="columns":
            log_string = f"Column '{c}' dtype: {df[c].dtype} -> "
        
        # if column is categorical
        if c in categorical_cols:
            df[c] = df[c].astype("category")
            if verbose=="columns":
                print(log_string + str(df[c].dtype))
            continue
            
        IsInt = False
        mx = df[c].max()
        mn = df[c].min()

        # test if column can be converted to an integer
        try:
            asint = df[c].fillna(0).astype(np.int64)
            result = (df[c] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
#       If column is string
        except (ValueError, TypeError) as e:
            if verbose=="columns":
                print(log_string + str(df[c].dtype))
            continue
        
        
        if IsInt and df[c].isna().sum()>0:
            for float_precision in float_precisions:
                if float_precision[0] <= mn and mx <= float_precision[1]:
                    df[c] = df[c].astype(float_precisions[float_precision])
                    break
        elif IsInt and np.isfinite(mn) and np.isfinite(mx):
            for int_range in int_ranges:
                if int_range[0] <= mn and mx <= int_range[1]:
                    df[c] = df[c].astype(int_ranges[int_range])
                    break
        else:
            df[c] = df[c].astype("float16")
            
        if verbose=="columns":
            print(log_string + str(df[c].dtype))
    
    if verbose != "none":
        print(f"Memory usage of df before: {start_size} MB")
        print(f"Memory usage of df after: {df.memory_usage(deep=True).sum()//(1024)**2} MB")
        
    return df