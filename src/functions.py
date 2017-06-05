import json
import pandas as pd
import spacy
from pandas.io.json import json_normalize

def import_json_to_df(path):
    """
    Import file with list of JSON into a DataFrame.
    
    INPUT
    path : string
        Path to JSON file

    OUTPUT
    json_list : DataFrame
        'json_normalize' flattens the nested dicts in the json into new columns 
        and turns everything into one DataFrame object.

    """
    json_list = []
    
    with open(path, 'r') as f:
        for line in f:
            json_line = json.loads(line)

            # '__reference' is Datalinq internal and can be ignored (only in Facebook data)
            col_names = {'__reference', '__location', '_id', 'can_post'}
            for n in col_names:
                json_line.pop(n, None)

            json_list.append(json_line)
            
    return json_normalize(json_list, sep='_')

def apply_nlp_to_column(df, input_col='description'):
    """
    Pass in the dataframe and the column name to which NLP processing should be 
    applied. This returns a pandas series of the objects created.

    INPUT
    df : pandas dataframe
    input_col : string
        The name of the column to process

    OUTPUT
    Pandas series
        NLP objects created from the text fields as a pandas series, with indexing to
        match the original dataframe.
    """
    #NOTE: This is using English language models on mostly Dutch text, new models should be built
    
    input_index = df[df[input_col].notnull()].index
    nlp = spacy.load('en') # this assumes `python -m spacy download en` has been run after installing Spacy

    nlp_gen = nlp.pipe(df[df[input_col].notnull()][input_col], n_threads=6)

    return pd.Series(nlp_gen, index=input_index)

