Folder Structure:

There is duplicate and unused data in these folders because we had no time to clean everything up.

data:
	processed_data:
		11142017M:
			out.csv
			result.csv
			test.csv
			train.csv
		11152017M:
		...
		out.csv
		test.csv
		train.csv
	raw_data:
		11142017M:
			no_location_records.json
			result.csv
			screen_name.txt
			stream_output.json
			user_tweets.json
		11152017M:
		...

raw_data: 
Contains folders with raw data files marked by date and first initial of the group member who collected it (this is because "M" tweets were collected from the LA area and "R" tweets from the SF area)

each data folder:
	no_location_records.json: Contains raw tweet json from Twitter API with no location records, which was not used in our project.

	screen_name.txt: Unused.

	stream_output.json: Raw tweet json objects returned from the Twitter API. These json files are processed into our training and test data.

	user_tweets.json: Unused.

	result.csv: Contains data resulting from the first preprocessing step of parsing the raw json in to csv with columns corresponding to our selected features as well as the tweet text used to extract training data.


processed_data:
Contains folders with the same date structure as raw_data, as well as a single out.csv, train.csv, and test.csv containing all training and test data combined into single files.

each data folder:
	result.csv: Duplicate of result.csv from raw_data folders.

	test.csv: Unlabelled test data output for each date.

	out.csv: Data from test.csv with approximate labels assigned by the bag_of_words code, used for approximate evaluation of learning model results.

	train.csv: Labelled training data extracted from result.csv using the emoji polarity method.