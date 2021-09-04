import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

def word_counts(
    df:pd.DataFrame, 
    free_text_col:str, 
    out_prefix:str,
    words_to_keep:int=0,
    min_count:int=0):
    
    """
    Creates bag of words vectors for a free text column

    Args:
        df: dataframe containing the free text column.
        free_text_col: name of the free_text_column.
        out_prefix: prefix of word count columns in output dataframe
        words_to_keep: 
            number of most frequent words to ouput
            Note, if words_to_keep is not specified, then all words with cummulative count of 2 or more are output

    Returns: 
        out: input dataframe appeneded with vector for bag of words
        bow_reference: dataframe with bag of words index
    """
    
    vectorisor = CountVectorizer(strip_accents="unicode", lowercase=True)
    x = vectorisor.fit_transform(df[free_text_col])
    counts = x.sum(axis=0).tolist()[0]
    words = vectorisor.get_feature_names()

#     Creates a df from bow index, words and count
    bow_reference = (pd.DataFrame(zip(words, counts), columns=["words", "counts"])
        .reset_index()
        .rename(columns={"index":"bow_key"})
        .sort_values("counts",ascending=False))
    if words_to_keep==min_count==0:
        min_count = 2
    elif min_count==0:
        min_count = bow_reference.iloc[words_to_keep]["counts"]
    elif words_to_keep > bow_reference.shape[0]:
        min_count = -1
    else:
        min_count = min(min_count, bow_reference.iloc[words_to_keep]["counts"])
        
    bow_reference["top_n"] = bow_reference["counts"] >= min_count

    #     appends top_n word columns to df
    keys = bow_reference[bow_reference["top_n"]]["bow_key"]
        
    out = df.merge(
        pd.DataFrame(
            x[:, keys].toarray(),
            columns=[f"{out_prefix}{key}" for key in keys]),
        left_index=True,
        right_index=True)
    
    return out, bow_reference[bow_reference["top_n"]]
