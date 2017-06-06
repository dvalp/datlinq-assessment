import json
import numpy as np
import pandas as pd
import spacy
import string
from pandas.io.json import json_normalize
from sklearn.feature_extraction.text import TfidfVectorizer


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

    #NOTE: `n_threads=-1` causes an out of memory error
    nlp_gen = nlp.pipe(df[df[input_col].notnull()][input_col], n_threads=6)

    return pd.Series(nlp_gen, index=input_index)

def find_other_documents(ref_document, df, compare_column='description_nlp', least_similar=False, num_items=10):
    """
    Return either the most similar or least similar texts based on their cosine similarity. First create 
    an Index based on the level of similarity, then return a slice of the DataFrame with index, name, and
    description.

    INPUT
    ref_document : SpaCy Doc type
        Any Spacy object that has `.similarity()` as a method
    df : pandas dataframe
        Any dataframe containing a column of SpaCy objects with vectors for similarity conparison
    compare_column : string
        Name of the column of objects to compare with ref_document
    least_similar : boolean
        False causes series to sort in descending order
    num_items : int
        Number of results to return

    OUTPUT
    Pandas DataFrame
        Return a slice of a DataFrame with index number, item name, and item description.
    """
    #NOTE: Uses English models on Dutch text which damage the accuracy of the results, new models should be built

    # Create an index of documents ordered by level of similarity to the reference
    idx_docs = df[df[compare_column].notnull()][compare_column] \
            .apply(ref_document.similarity) \
            .sort_values(ascending=least_similar)[1:num_items + 1] \
            .index

    return df.loc[idx_docs][['name', 'description']]

def tokens_from_spacy(doc):
    """
    Takes a SpaCy Doc object rather than a normal string. Stores any words longer than 4 
    characters and not standard punctuation in a list of strings (tokens).

    INPUT
        doc : SpaCy Doc
            requires the Doc to have words and .lemma_

    OUTPUT
        tokens : list(str)
            returns tokens as a list of strings
    """
    tokens = []

    for word in doc:
        if word.lemma_ == '-PRON-':
            tokens.append(word.string.lower())
        elif (word.lemma_.strip() not in string.punctuation) and (len(word.lemma_.strip()) > 3):
            tokens.append(word.lemma_)

    return ' '.join(tokens)

def spacy_docs_to_df_tfidf(df, doc_col='description_nlp', max_df=0.9, min_df=10, max_features=1000):
    tfidf_vec = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features, stop_words='english')
    tfidf = tfidf_vec.fit_transform(
            df[df[doc_col].notnull()][doc_col]
            .apply(tokens_from_spacy))
    idx_tfidf = df[df[doc_col].notnull()].index

    return pd.SparseDataFrame(tfidf, index=idx_tfidf), np.array(tfidf_vec.get_feature_names())

def top_n_doc_tokens(doc_idx, df_tfidf, vocab, max_tokens=10):
    return vocab[df_tfidf.loc[doc_idx].sort_values(ascending=False).dropna().index][:max_tokens]

