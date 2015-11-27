# code for getting entity-label data.

from collections import defaultdict
import codecs

# custom imports
import sparql_endpoint


class Labels(object):
	def __init__(self):
		self.data = defaultdict(set)

	def Add_label_data(self, label, resource_uri):
		self.data[label].add(resource_uri)

	def keys(self):
		return self.data.keys()

	def iteritems(self):
		return self.data.items()


# gets label data from the given endpoint.
def Get_from_endpoint(endpoint_url, graph_iri, query_file):
	with open(query_file) as h:
		query = h.read().strip()

	# retrieve labels.
	print("Retrieving labels from SPARQL endpoint.")
	uri_label_data = sparql_endpoint.Query(endpoint_url, query, graph_iri)

	labels = Labels()
	for row in uri_label_data:
		if len(row['uri']) > 0 and len(row['label']) > 0:
			labels.Add_label_data(row['label'], row['uri'])

	return labels


# gets label data from the given text file.
def Get_from_file(filename):
	print("Retrieving labels from cached file.")
	labels = Labels()
	with codecs.open(filename, encoding='utf-8') as h:
		for line in h:
			tokens = line.rstrip().split('\t')
			if len(tokens) != 2:
				continue

			uri, label = tokens
			if len(uri) == 0 or len(label) == 0:
				continue

			labels.Add_label_data(label, uri)

	return labels

def Write_to_file(labels, filename):
	print("Writing labels to", filename, ".")

	with codecs.open(filename, 'w', encoding='utf-8') as h:
		for label, uris in labels.iteritems():
			for uri in uris:
				h.write(uri + u'\t' + label + u'\n')
