# Facebook data

## Origin

Data from facebook is obtained by valid key interfacing with the Facebook API.
Data is stored in a local MongoDB by a script running every minute of every day.

## Original Data

All fields, not prefixed by underscore, are stored as is in the json.
All possible fields are described in:

 * <https://developers.facebook.com/docs/graph-api/reference/location/>
 * <https://developers.facebook.com/docs/graph-api/reference/page/>
 * <https://developers.facebook.com/docs/graph-api/reference/place>
 
We combine all fields from different requests into one 'location' object
Do note that not very location has all possible fields.

This data set should contain no duplicate facebook id's

## Added data

We have also added data, identified by fields beginning with underscores for more effective processing or storing meta data

* **__location** : MongoDB coordinate object for geospatial queries, based on lat & long in original data
* **__reference** : Datrlinq internal, can be ignored
* **__timestamp** : Overal UNIX timestamp of the record
* **__timestamp_basic/detail/etc** : Most recent Unix timestamp of each of the subqueries used for creating this object
* **_id** : Mongo ID field is also facebook unique ID


## Notes

Due to export from Mongo, some type safety (like Long numbers) are converted in mongo json, 
eg:
 `"talking_about_count" : { "$numberLong" : "1345" }` instead of `"talking_about_count" : 1345`


