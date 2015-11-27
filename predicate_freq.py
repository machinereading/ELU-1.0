# code for getting link predicate frequency data.

from collections import defaultdict
import codecs

# custom imports
import sparql_endpoint


# gets predicate frequency data from the given endpoint.
def Get_from_endpoint(endpoint_url, graph_iri, query_file):
	with open(query_file) as h:
		query = h.read().strip()

	# retrieve labels.
	print("Retrieving predicate frequency data from SPARQL endpoint.")
	pred_freq_data = sparql_endpoint.Query(endpoint_url, query, graph_iri)

	pred_freq = defaultdict(int)
	for row in pred_freq_data:
		if len(row['pred']) > 0:
			pred_freq[row['pred']] = int(row['freq'])

	return pred_freq

# gets predicate frequency data from the given text file.
def Get_from_file(filename):
	print("Retrieving predicate frequency data from cached file.")
	pred_freq = defaultdict(int)
	with codecs.open(filename, encoding='utf-8') as h:
		for line in h:
			tokens = line.rstrip().split('\t')
			if len(tokens) != 2:
				continue

			uri, freq = tokens
			if len(uri) == 0 or len(freq) == 0:
				continue

			pred_freq[uri] = int(freq)

	return pred_freq

def Write_to_file(pred_freq, filename):
	print("Writing predicate frequency data to", filename, ".")

	with codecs.open(filename, 'w', encoding='utf-8') as h:
		for pred, freq in pred_freq.iteritems():
			h.write(pred + u'\t' + unicode(freq) + u'\n')
