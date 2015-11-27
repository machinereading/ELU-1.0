import re
from itertools import product
from collections import defaultdict

# custom imports
import sparql_endpoint


class Config:
	endpoint_url = None
	graph_iri = None
	relation_query = None

	@staticmethod
	def Init(config):
		with open(config['domains'][0]['relation_query']) as h:
			Config.relation_query = h.read().strip()

		Config.endpoint_url = config['domains'][0]['endpoint']

		if 'graph_iri' in config['domains'][0]:
			Config.graph_iri = config['domains'][0]['graph_iri']


class Relations(object):
	_relations_cache = {}
	def __init__(self, tokens, candidates):
		self._tokens = tokens
		self._candidates = candidates
		self.relations = {}
		self.total_requests = 0
		self.hit_requests = 0

	def Related_candidate_uris(self, target_candidate, target_candidate_uri, candidates):
		uri_relations = {}

		for candidate in target_candidate.neighbor_candidates:
			for candidate_uri in candidate.uris:
				if candidate_uri.is_candidate:
					relations = self._Get_entity_relations(target_candidate_uri.uri, candidate_uri.uri)
					if len(relations) > 0:
						uri_relations[candidate_uri.uri_id] = relations

		return uri_relations


	@staticmethod
	def _Entities_key(entity1, entity2):
                
		return tuple(sorted([entity1, entity2]))

	def _Get_entity_relations(self, entity1, entity2):
		key = Relations._Entities_key(entity1, entity2)
		self.total_requests += 1

		if key not in Relations._relations_cache:
			Relations._relations_cache[key] = _Entity_relations(entity1, entity2)
		else:
			print("hello")
			self.hit_requests += 1

		return Relations._relations_cache[key]


def Get_relations(tokens, candidates):
	return Relations(tokens, candidates)

def _Entity_relations(entity1, entity2):
	query = _Relations_query(entity1, entity2)
	results = sparql_endpoint.Query(Config.endpoint_url, query, Config.graph_iri)
	return [r['pred'] for r in results]

def _Relations_query(entity1, entity2):
	# the entities have to be encoded strings, for the query to work.
	#encoded_entity1 = _Get_encoded_string(entity1)
	#encoded_entity2 = _Get_encoded_string(entity2)

	query = re.sub(r'\[\[X1\]\]', "<" + entity1 + ">", Config.relation_query)
	query = re.sub(r'\[\[X2\]\]', "<" + entity2 + ">", query)
	return query

def _Get_encoded_string(s):
	if isinstance(s, str):
		return s.encode('utf-8')
	else:
		return s


if __name__ == "__main__":
	sys.exit("use routing.py")
