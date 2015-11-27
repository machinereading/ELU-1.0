from collections import defaultdict
import re

# custom imports
import sparql_endpoint


class Config:
	endpoint_url = None
	graph_iri = None
	object_score_query = None
	subject_score_query = None

	@staticmethod
	def Init(config):
		with open(config['domains'][0]['subject_score_query']) as h:
			Config.subject_score_query = h.read().strip()

		with open(config['domains'][0]['object_score_query']) as h:
			Config.object_score_query = h.read().strip()

		Config.endpoint_url = config['domains'][0]['endpoint']

		if 'graph_iri' in config['domains'][0]:
			Config.graph_iri = config['domains'][0]['graph_iri']


class Resource_base_scores(object):
	_scores_cache = {}
	def __init__(self, matches):
		# self._scores_cache = {}
		self.scores = {}

		for match in matches:
			for uri in match.uris:
				scores = self.Entity_scores(uri['uri'])
				self.scores[uri['id']] = scores

	def Scores_of_uri_id(self, uri_id):
		return self.scores[uri_id]

	def Entity_scores(self, entity):
		if entity not in Resource_base_scores._scores_cache:
			Resource_base_scores._scores_cache[entity] = _Entity_score(entity)

		return Resource_base_scores._scores_cache[entity]


def Entity_base_scores(matches):
	return Resource_base_scores(matches)

def _Entity_score(entity_uri):
	# subject_query = _Relations_query(Config.subject_score_query, entity_uri)
	# subject_results = sparql_endpoint.Query(Config.endpoint_url, subject_query, Config.graph_iri)

	object_query = _Relations_query(Config.object_score_query, entity_uri)
	object_results = sparql_endpoint.Query(Config.endpoint_url, object_query, Config.graph_iri)

	relation_counts = {}
	for row in object_results:
		relation_counts[row['p']] = int(row['count'])

	return relation_counts

	# return (int(subject_results[0]['count']), int(object_results[0]['count']))

def _Relations_query(query_template, entity):
	# the entities have to be encoded strings, for the query to work.
	#encoded_entity = _Get_encoded_string(entity)
	#print(encoded_entity)

	query = re.sub(r'\[\[X1\]\]', "<" + entity + ">", query_template)
	return query

def _Get_encoded_string(s):
	if isinstance(s, str):
		return s.encode('utf-8')
	else:
		return s
