import scipy.io
import numpy as np
import pandas as pd
import xml.dom.minidom as dom
import time
import datetime

def load_sed2012_dataset(subset_size=10000, sort_by_uploaded=True, event_types=False, binary=False, noise_rate=0.95):
	# File paths for metadata and ground truth per challenge
	metadata_file = "dataset/sed2012/sed2012_metadata.xml"
	technical_file = "dataset/sed2012/technical_events.txt"
	soccer_file = "dataset/sed2012/soccer_events.txt"
	indignados_file = "dataset/sed2012/indignados_events.txt"

	# Create ground truth mapping (photo ID -> event ID)
	ground_truth = {}

	if binary:
		event_types = True

	with open(technical_file, "r") as f:
		create_array(f.readlines(), ground_truth)
	min_technical = 1
	max_technical = max(ground_truth.values())
	with open(soccer_file, "r") as f:
		create_array(f.readlines(), ground_truth, class_counter=max(ground_truth.values()) + 1)
	min_soccer = max_technical + 1
	max_soccer = max(ground_truth.values())
	with open(indignados_file, "r") as f:
		create_array(f.readlines(), ground_truth, class_counter=max(ground_truth.values()) + 1)
	min_indignados = max_soccer + 1
	max_indignados = max(ground_truth.values())

	# Parse metadata and create DataFrame
	df = get_modalities(ground_truth, metadata_file)

	if binary:
		df['event_type'] = df['event_id'].apply(
		lambda eid: 1 if min_technical <= eid <= max_indignados else
					0  # Default to 0 for unknown/other
		)
	else:
		df['event_type'] = df['event_id'].apply(
		lambda eid: 1 if min_technical <= eid <= max_technical else
					2 if min_soccer <= eid <= max_soccer else
					3 if min_indignados <= eid <= max_indignados else
					0  # Default to 0 for unknown/other
		)
	
	# Ground truth labels
	labels = df['event_type'].to_numpy() if event_types else df['event_id'].to_numpy()
	print(f"Unique labels: {np.unique(labels)}")

	subset_size = min(subset_size, len(df))

	rng = np.random.default_rng(0)

	if 0 <= noise_rate < 1.0:
		print(noise_rate)
		# Split noise (labels == 0) and event (labels > 0)
		noise_indices = np.where(labels == 0)[0]
		event_indices = np.where(labels > 0)[0]
		print(f"total amount of event indices: {len(event_indices)}")

		num_events = min(int((1-noise_rate) * subset_size), len(event_indices))
		num_noise = subset_size - num_events
	
    	# Randomly sample noise and event indices
		sampled_noise_indices = rng.choice(noise_indices, num_noise, replace=False)
		sampled_event_indices = rng.choice(event_indices, num_events, replace=False)

		# Combine sampled indices and shuffle
		sampled_indices = np.concatenate([sampled_noise_indices, sampled_event_indices])
		sampled_indices = np.sort(sampled_indices)

		# Subset the data and labels
		df = df.iloc[sampled_indices]

		# Get the labels again
		labels = df['event_type'].to_numpy() if event_types else df['event_id'].to_numpy()

	if sort_by_uploaded:
		df = df.sort_values(by='dateupload')

	# Modality 1: Time data (date taken and uploaded)
	df['datetaken'] = df['datetaken'].replace(['0000-00-00 00:00:00'], '1970-01-01 00:00:00').apply(convertToTimestamp)
	df['dateupload'] = df['dateupload'].replace(['0000-00-00 00:00:00'], '1970-01-01 00:00:00').apply(convertToTimestamp)
	time_modality = df[['datetaken', 'dateupload']].to_numpy()

	# Modality 2: Geospatial data (latitude and longitude)
	location_modality = df[['latitude', 'longitude']].fillna(-1).to_numpy()

	# Modality 3: Text data / Tags (placeholder: tag count)
	tags_modality = df['tag'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0).to_numpy().reshape(-1, 1)

	# Sanity check to ensure alignment
	assert time_modality.shape[0] == location_modality.shape[0] == tags_modality.shape[0] == labels.shape[0], "Mismatch in number of samples between modalities and labels"

	# Return modalities and labels
	return [location_modality, time_modality, tags_modality], ["location", "time", "tags"], labels

def create_array(lines, ground_truth, class_counter=1):
	arr = []
	counter = class_counter
	for line in lines:
		split = line.split(",")
		split[-1] = split[-1][:-1]
		for e in split:
			ground_truth[e] = counter
		counter += 1
		arr.extend(split)
	for el in arr:
		if len(el) != 10:
			arr.remove(el)
	return arr

def get_modalities(ground_truth, metadata_path):
	xml = dom.parse(metadata_path)
	photos = xml.getElementsByTagName("photo")
	A = []
	for photo in photos:
		id = photo.getAttributeNode("id").nodeValue
		if id in ground_truth.keys():
			_class = ground_truth[id]
		else:
			_class = 0

		datetaken = photo.getAttributeNode("dateTaken").nodeValue
		dateupload = photo.getAttributeNode("dateUploaded").nodeValue

		try:
			location = photo.getElementsByTagName("location")[0]
			lattitude = float(location.getAttributeNode("latitude").nodeValue)
			longitude = float(location.getAttributeNode("longitude").nodeValue)
		except:
			lattitude = -1
			longitude = -1

		try:
			tags = [tag.firstChild.data for tag in photo.getElementsByTagName("tag")]
			tags = " ".join(tags)
		except:
			tags = ""

		try:
			title = photo.getElementsByTagName("title")[0].firstChild.nodeValue
		except:
			title = ""

		try:
			description = photo.getElementsByTagName("description")[0].firstChild.nodeValue
		except:
			description = ""

		user = photo.getAttributeNode("dateTaken").nodeValue

		A.append([id, datetaken, dateupload, lattitude, longitude, title, description, tags, user, _class])

	df = pd.DataFrame(A, columns=['flickr_picture_id', 'datetaken', 'dateupload', 'latitude', 'longitude', 'title', 'description', 'tag', 'username', 'event_id'])

	df['flickr_picture_id'] = df['flickr_picture_id'].astype(int)
	return df

def convertToTimestamp(x):
    return time.mktime(datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S.%f").timetuple())

def load_synthetic_dataset(subset_size=None):
	file_path = "swfd/dataset/synthetic_n=500000,m=10,d=300,zeta=10.mat"
	data = scipy.io.loadmat(file_path)["A"]
	if subset_size is not None and subset_size > 0 and subset_size < len(data):
		data = data[:subset_size]
	return [data.astype(np.float64)]
