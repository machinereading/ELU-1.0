import urllib
import json
import re
from time import sleep

class Constants:
	SPARQL_LIMIT = 10000


# Retrieves results of a SPARQL query from an endpoint.
# This function doesn't support LIMITS. All (even 10000+) results are retrieved.
# All query arguments are present as keys in all rows of the resulting array. (Null column == None value)
# All results (and their keys) are in Unicode.
# query_str: not url_encoded
def Query(endpoint_url, query_str, graph_iri='', timeout=0, delay=0):
	args = {
		'default-graph-uri': graph_iri,
		'format': 'application/json',
		'query': None,
		'timeout': timeout
	}

	current_offset = 0
	result_data = []

	while(True):
		if current_offset > 0:
			print(current_offset)
		args['query'] = query_str + ' LIMIT ' + str(Constants.SPARQL_LIMIT) + ' OFFSET ' + str(current_offset)
		data = urllib.parse.urlencode(args)
		url = endpoint_url + '?' + data
		request = urllib.request.urlopen(url)

		response_str = request.read()
		response_data = None

		try:
			response_data = json.loads(response_str)
		except Exception as e:
			response_str = re.sub(r'\\U[0-9A-Za-z]{8}', '', response_str.decode())
			response_data = json.loads(response_str)
			

		query_vars = response_data['head']['vars']

		for row in response_data['results']['bindings']:
			result_row = {}
			for var in query_vars:
				if var in row:
					result_row[var] = row[var]['value']
				else:
					result_row[var] = None

			result_data.append(result_row)
			
		if len(response_data['results']['bindings']) == Constants.SPARQL_LIMIT:
			current_offset += 10000
		else:
			break

	if delay > 0:
		# sleep a bit to preserve endpoint sanity.
		sleep(delay)

	return result_data


if __name__ == "__main__":
	print("sparql_endpoint Module: import to use!")
