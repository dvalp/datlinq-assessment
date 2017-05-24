# Factual data

## Origin

Data from factual is obtained by a contact between Datlinq & Factual and a monthly data dump.
Data is stored in a local MariaDB by a script running every month.

## Original Data

All fields, not prefixed by underscore, are stored as is in the csv.
Some records are serialized json objects (category_ids, category_labels, hours, neigborhood, etc)

All possible fields are queryable via the API:

 * <http://developer.factual.com/v3/>
 

Catgeories are described in 
 
 * <http://developer.factual.com/working-with-categories/>
 
 Existence score is defined: 
 
* <http://developer.factual.com/search-place-rank-and-existence/>

This dataset should contain no duplicates

## Added data

We have also added data, identified by fields beginning with underscores for more effective processing or storing meta data

* **_org_filename** : Original import filename
* **_org_filedate** : Original import timestamp of the file (not when it was imported)
* **_imported** : Timestamp of when data was imported
* **__main_category_id** : Extracted first id from **category_ids** field
* **__hash** : Generated hash used for identifying  data mutations since last import


## Notes




