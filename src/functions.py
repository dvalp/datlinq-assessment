import json
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

