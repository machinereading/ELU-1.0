from __future__ import division

import json
import traceback
import re

# custom imports
import sparql_endpoint
import exact_surface_matching
import entity_labels
import predicate_freq
import entity_relations
import entity_base_scoring
import output
import features
import sys

# main configuration
class Config:
	endpoint = None
	graph_iri = None
	morph_analysis = None
	use_svm = None
	ignore_numbers = None
	relation_word_limit = 10

	@staticmethod
	def Init(config):
		Config.endpoint = config['domains'][0]['endpoint']
		Config.graph_iri = config['domains'][0]['graph_iri']

		# set the language (and tagger).
		# new taggers require new modules to be implemented.
		tagger = config['lang']
		if tagger == "ko-etri":
			import morph_analysis_etri
			Config.morph_analysis = morph_analysis_etri
		elif tagger == "ja-mecab":
			import morph_analysis_mecab
			Config.morph_analysis = morph_analysis_mecab
		else:
			raise Exception(tagger + ": non-valid tagger string literal.")

		Config.use_svm = config['use_svm']
		Config.ignore_numbers = config['ignore_numbers']

# the base class & functions used during the main EL process.
# all entity linking functionality classes inherit this class.
class Entity_detection_functions(object):
	def __init__(self, tokens, matches, uri_base_scores):
		self.tokens = tokens
		self.matches = matches
		self.uri_base_scores = uri_base_scores
		self.entity_candidates = [Entity_candidate(self, candidate_id, tokens, match) for candidate_id, match in enumerate(matches)]
		self.relations = None
		self.comments = {}
		self.uri_id_map = self._Uri_id_mapping()

	# create a mapping of uri ids => uris.
	def _Uri_id_mapping(self):
		uri_id_map = {}

		for candidate in self.entity_candidates:
			for uri in candidate.uris:
				uri_id_map[uri.uri_id] = uri

		return uri_id_map

	# create a mapping of which neighbor candidates each candidate has.
	def _Candidate_neighbor_mapping(self):
		current_candidates_in_range = []
		next_candidates_in_range = []
		
		for candidate in sorted(self.entity_candidates, key=lambda c:(c.start_token_id, -1*c.end_token_id)):
			# check if each target candidate in range is a neighbor of this candidate.
			for target_candidate in current_candidates_in_range:

				# we check for the range condition first.
				# if this fails, the target candidate cannot have any more neighbors.
				if self.tokens[candidate.end_token_id].word_id >= self.tokens[target_candidate.start_token_id].word_id - Config.relation_word_limit and self.tokens[candidate.start_token_id].word_id <= self.tokens[target_candidate.end_token_id].word_id + Config.relation_word_limit:

					# add the target candidate to the list of candidates in range.
					next_candidates_in_range.append(target_candidate)
					
					# check for overlapping of position and uri.
					if candidate.id != target_candidate.id and not target_candidate.Overlaps_with(candidate) and not target_candidate.Meaning_overlaps_with(candidate):
						# these candidates are neighbors of each other.
						candidate.Add_neighbor_candidate(target_candidate)
						target_candidate.Add_neighbor_candidate(candidate)

			# add this candidate to the list of candidates in range.
			next_candidates_in_range.append(candidate)

			current_candidates_in_range = next_candidates_in_range
			next_candidates_in_range = []

	# assigns initial scores (everything except relation weights) to candidate entity uris.
	# this is used for learning.
	def _Candidate_uri_initial_scoring(self):
		self._Candidate_uri_scoring(set_representive_uri=True)


	# assigns entity scores to candidate entity uris.
	def _Candidate_uri_scoring(self, set_representive_uri=False, for_actual_detection=False):
		for candidate in self.entity_candidates:
			for uri in candidate.uris:
				uri.uri_score = features.Features.URI_initial_score(candidate, uri)
			
			if for_actual_detection:
				# only consider URIs that are at least comparable to the maximum one.
				max_uri = max(candidate.uris, key=lambda uri:uri.uri_score)
				for consider_uri in candidate.uris:
					if consider_uri.uri_score + 1 >= max_uri.uri_score:
						consider_uri.is_candidate = True

			if set_representive_uri:
				# set the most popular uri for each candidate as candidate uris.
				candidate.representive_uri = max(candidate.uris, key=lambda uri:uri.uri_score)
				candidate.representive_uri.is_candidate = True

	# def _Candidate_uri_baseline_scoring(self):
	# 	for candidate in self.entity_candidates:
	# 		for uri in candidate.uris:
	# 			uri.uri_score = features.Features.URI_initial_score(candidate, uri)
	# 			uri.score = uri.uri_score

	# 		candidate.representive_uri = max(candidate.uris, key=lambda uri:uri.uri_score)

	# discover relations between candidate uris.
	def _Assign_candidate_relations(self):
		self.relations = entity_relations.Get_relations(self.tokens, self.entity_candidates)

		for candidate in self.entity_candidates:
			for uri in candidate.uris:
				if uri.is_candidate:
					uri.relations = self.relations.Related_candidate_uris(candidate, uri, self.entity_candidates)

			# save the number of candidates that are near each candidate.
			candidate.neighbor_candidate_count = len(candidate.neighbor_candidates)

	# assign relation scores to candidate uris.
	def _Assign_uri_relation_scores(self):
		for candidate in self.entity_candidates:
			for uri in candidate.uris:
				uri.entity_relation_score = uri.Related_uri_score_sum()

	# chooses seed entity uri candidates to get relations between.
	def _Select_candidates(self):
		word_count = self.tokens[-1].word_id + 1
		entity_count_limit = features.Features.Seed_entity_count(word_count)
		relation_ratio_limit = features.Features.Seed_relation_ratio()

		chosen_entity_count = 0

		end_by_entity_count = False
		end_by_relation_ratio = False

		final_entity_ratio_limit = None
		final_relation_ratio_limit = None


		for candidate in sorted(self.entity_candidates, key=lambda c:c.Learning_scores(), reverse=True):
			overlaps = False

			# for compare_candidate in self.entity_candidates:
			# 	if compare_candidate.is_candidate and compare_candidate.Overlaps_with(candidate):
			# 		overlaps = True
			# 		break

			if not overlaps:
				candidate.chosen_for_learning = True
				chosen_entity_count += 1
				# self._limit_uri_base_score = candidate.initial_score

				if not end_by_entity_count and not end_by_relation_ratio:
					# check for initial terminal conditions.
					if chosen_entity_count >= entity_count_limit:
						end_by_entity_count = True
						final_relation_ratio_limit = features.Features.Final_seed_relation_ratio_limit(candidate.Learning_scores())

					elif candidate.Learning_scores() < relation_ratio_limit:
						end_by_relation_ratio = True
						final_entity_ratio_limit = features.Features.Final_seed_entity_ratio_limit(word_count / chosen_entity_count)

				# check for final terminal conditions.
				if (end_by_entity_count and candidate.Learning_scores() <= final_relation_ratio_limit) or (end_by_relation_ratio and (word_count / chosen_entity_count) <= final_entity_ratio_limit):
					features.Features.Update_ratio_limits(word_count / chosen_entity_count, candidate.Learning_scores())
					break

	# determine which candidates are entities, and their respective uris.
	def _Select_entities(self, lower_bound):
		for candidate in sorted(self.entity_candidates, key=lambda c:(c.representive_score, -1 * c.start_token_id, c.end_token_id), reverse=True):
			# if the system is configured to ignore numbers, ignore them.
			if Config.ignore_numbers and candidate.text.isdigit():
				continue

			if candidate.representive_score < lower_bound:
				
				break

			overlaps = False

			for compare_candidate in self.entity_candidates:
				compare_candidate.contained_by_potential_entity = False

				if compare_candidate.is_entity:
					if candidate.Contains(compare_candidate):
						compare_candidate.contained_by_potential_entity = True
					elif candidate.Overlaps_with(compare_candidate):
						overlaps = True
						break

			if not overlaps:
				candidate.is_entity = True

				# undesignate all entites that are contained within this entity.
				for compare_candidate in self.entity_candidates:
					if compare_candidate.contained_by_potential_entity:
						compare_candidate.is_entity = False
						compare_candidate.contained_by_potential_entity = False
						compare_candidate.selected_uri = None

				candidate.selected_uri = max(candidate.uris, key=lambda uri:uri.score)
				#print(candidate.text, ' --> ',candidate.selected_uri.uri, ' Score : ', candidate.selected_uri.score)
				# self._limit_uri_base_score = candidate.initial_score

	# def _Select_baseline_entities(self):
	# 	for candidate in sorted(self.entity_candidates, key=lambda c:c.representive_uri.uri_score, reverse=True):
	# 		if candidate.Is_baseline_entity():
	# 			overlaps = False

	# 			for compare_candidate in self.entity_candidates:
	# 				if compare_candidate.is_entity:
	# 					if candidate.Overlaps_with(compare_candidate):
	# 						overlaps = True
	# 						break

	# 			if not overlaps:
	# 				candidate.is_entity = True
	# 				candidate.selected_uri = candidate.representive_uri


	# assign scores to entity candidates.
	def _Assign_entity_scores(self, for_actual_detection=False):
		for candidate in self.entity_candidates:
			for uri in candidate.uris:
				uri.score = features.Features.URI_score(candidate, uri)
				#print(uri.uri, ' --> ' ,uri.score)
				
			if for_actual_detection:
				for uri in candidate.uris:
					if Config.use_svm:
						uri.entity_confidence = uri.score
					else:
						uri.entity_confidence = features.Features.afterscore.Entity_confidence(uri)
				candidate.representive_score = max([u.entity_confidence for u in candidate.uris])
	
	# return ordered & filtered entity candidates for output purposes.
	def Candidates_for_output(self):
		return sorted(self._Remove_candidate_overlaps(), key=lambda c:c.start_token_id)

	# for output: remove candidates that overlap with each other (position-wise)
	# so that we can represent the text in non-overlapping chunks of characters.
	def _Remove_candidate_overlaps(self):
		trimmed_candidates = list(filter(lambda c:c.is_entity, self.entity_candidates))

		for candidate in sorted(self.entity_candidates, key=lambda c:(c.start_token_id,c.end_token_id*-1)):
			if not candidate.is_entity:
				overlap = False
				for trimmed_candidate in trimmed_candidates:
					if trimmed_candidate.Overlaps_with(candidate):
						overlap = True
						break

				if not overlap:
					trimmed_candidates.append(candidate)

		return trimmed_candidates



# actual entity selection & scoring
class Entity_detection(Entity_detection_functions):
	def __init__(self, tokens, matches, uri_base_scores, lower_bound=0.5):
		Entity_detection_functions.__init__(self, tokens, matches, uri_base_scores)
		self.lower_bound = lower_bound

		print("Performing candidate uri scoring...")
		self._Candidate_uri_scoring(for_actual_detection=True)

		print("Finding candidate neighbors...")
		self._Candidate_neighbor_mapping()

		print("Finding candidate uri relations...")
		self._Assign_candidate_relations()

		print("Assigning scores to uris...")
		self._Assign_entity_scores(for_actual_detection=True)

		print("Assigning entity relation scores to uris...")
		self._Assign_uri_relation_scores()

		print("Selecting entities...")
		self._Select_entities(self.lower_bound)

		# self.comments['lower bound'] = features.Features.afterscore.Threshold()
		self.comments['lower bound'] = self.lower_bound
		self.comments['hit_requests'] = self.relations.hit_requests
		self.comments['total_requests'] = self.relations.total_requests

# class Baseline_entity_detection(Entity_detection_functions):
# 	def __init__(self, tokens, matches, uri_base_scores):
# 		Entity_detection_functions.__init__(self, tokens, matches, uri_base_scores)

# 		print "Performing candidate uri initial scoring..."
# 		self._Candidate_uri_baseline_scoring()

# 		print "Selecting entities..."
# 		self._Select_baseline_entities()
	

# entity linking training
class Entity_detection_learning(Entity_detection_functions):
	def __init__(self, tokens, matches, uri_base_scores):
		Entity_detection_functions.__init__(self, tokens, matches, uri_base_scores)

		# learn feature multipliers.
		# print "Learning base feature multipliers..."
		# features.Features.Learn_kb_features(self.tokens, self.entity_candidates, self.uri_base_scores)

		print("Performing candidate uri initial scoring...")
		self._Candidate_uri_initial_scoring()

		print("Finding candidate neighbors...")
		self._Candidate_neighbor_mapping()

		print("Finding relations between initial candidates...")
		self._Assign_candidate_relations()

		print("Learning relation weights...")
		features.Features.Learn_relation_weights(self.entity_candidates)

		print("Updating uri scores according to new relation weights...")
		self._Candidate_uri_scoring()

		print("Choosing candidates...")
		self._Select_candidates()

		print("Assigning scores to URIs...")
		self._Assign_entity_scores()

		# learn P(e|f) for each feature.
		print("Learning entity features...")
		features.Features.Learn_entity_features(self.entity_candidates)

		print("Saving entity features...")
		features.Features.Save_entity_features(self.entity_candidates)

		print("Learning entity afterscores...")
		features.Features.Learn_entity_afterscores(self.entity_candidates)

		# if use_svm:
		# 	# svm model: save features of entities / non-entities.
		# 	print "Saving entity features..."
		# 	features.Features.Save_entity_features(self.entity_candidates)
		# else:
		# 	# confidence model: learn afterscores for entites / non-entities.
		# 	print "Learning entity afterscores..."
		# 	features.Features.Learn_entity_afterscores(self.entity_candidates)

		# print "Learning POS features..."
		# features.Features.Learn_POS_features(self.entity_candidates)

		# print "Learning word length features..."
		# features.Features.Learn_word_length_features(self.entity_candidates)

		# comments
		self.comments['word count'] = self.tokens[-1].word_id + 1
		# self.comments['word length multiplier'] = features.Features.word_length_multiplier
		# self.comments['subject freq multiplier'] = features.Features.uri_subject_freq_multiplier
		self.comments['object freq multiplier'] = features.Features.uri_object_freq_multiplier

		pos_features_output = features.Features.pos.Output_repr()
		self.comments['feature:has pos'] = pos_features_output['has_pos']
		self.comments['feature:start pos'] = pos_features_output['start_pos']
		self.comments['feature:end pos'] = pos_features_output['end_pos']
		self.comments['feature:prev pos'] = pos_features_output['prev_pos']
		self.comments['feature:next pos'] = pos_features_output['next_pos']

		self.comments['feature:word length'] = features.Features.word_length.Output_repr()
		self.comments['feature:uri object'] = features.Features.uri_object.Output_repr()
		self.comments['hit_requests'] = self.relations.hit_requests
		self.comments['total_requests'] = self.relations.total_requests

		# with open('entity-uriscore.txt', 'a') as e_h, open('nonentity-uriscore.txt', 'a') as ne_h:
		# 	for candidate in self.entity_candidates:
		# 		if candidate.chosen_for_learning:
		# 			handle = e_h
		# 		else:
		# 			handle = ne_h

		# 		handle.write(str(candidate.representive_uri.uri_score) + '\n')


class Entity_candidate(object):
	def __init__(self, module, candidate_id, tokens, match):
		self.module = module
		self.id = candidate_id
		self.text = tokens.Text_of_token_offsets(match.start, match.end)
		self.pos = tokens.POS_of_token_offsets(match.start, match.end)
		self.uris = [Entity_candidate_uri(uri, self) for uri in match.uris]

		self.chosen_for_learning = False
		self.is_entity = False
		self.selected_uri = None
		self.base_score = None
		self.initial_score = 0.0

		self.neighbor_candidate_count = 0
		self.representive_uri = None
		self.representive_score = None
		# self.neighbor_entity_count = 0

		self.contained_by_potential_entity = False

		self.start_token_id = match.start
		self.end_token_id = match.end

		self.start_word_id = tokens[match.start].word_id
		self.end_word_id = tokens[match.end].word_id

		self.neighbor_candidates = []

		# get the POS of the previous token.
		# if there isn't a previous token or this candidate starts at the start of a word,
		# we assume BLANK.
		if self.start_token_id == 0 or tokens[self.start_token_id].is_word_start:
			self.prev_pos = 'BLANK'
		else:
			# we get the last POS tag of the previous token.
			self.prev_pos = tokens[self.start_token_id-1].pos[-1]

		# get the POS of the next token.
		# if there isn't a next token or this candidate ends at the end of a word,
		# we assume BLANK.
		if self.end_token_id >= len(tokens)-1 or tokens[self.end_token_id].is_word_end:
			self.next_pos = 'BLANK'
		else:
			# we get the first POS tag of the next token.
			self.next_pos = tokens[self.end_token_id+1].pos[0]

	# add the given candidate as a neighbor of this candidate.
	def Add_neighbor_candidate(self, candidate):
		self.neighbor_candidates.append(candidate)

	# def Is_baseline_entity(self):
	# 	return ((self.pos[0][0] == 'N' or self.pos[0] == 'SN') and self.pos[-1][0] == 'N' and self.prev_pos == "BLANK" and (self.next_pos == "BLANK" or self.next_pos[0] == "X" or self.next_pos[0] == "J"))

	# get the base score of this candidate.
	def Base_score(self):
		if self.base_score is None:
			self.base_score = max(uri.base_score for uri in self.uris)

		return self.base_score

	# return the score determining whether this candidate gets classified as an entity for training.
	def Learning_scores(self):
		return self._Relation_ratio()

	# get the relation ratio; a ratio representing how much this candidate is related to neighbors.
	def _Relation_ratio(self):
		# return len(self.representive_uri.relations.keys()) / self.neighbor_candidate_count
		if self.neighbor_candidate_count > 0:
			return self.representive_uri.Relation_weight_sum() / self.neighbor_candidate_count
		else:
			return 0

	# get the uri with the highest uri score.
	def Top_uri(self):
		if self.selected_uri is None:
			self.selected_uri = max(self.uris, key=lambda uri: uri.score)

		return self.selected_uri

	# check if this candidate position-wise overlaps with the given candidate.
	def Overlaps_with(self, candidate):
		return (not ((self.start_token_id > candidate.end_token_id) or (self.end_token_id < candidate.start_token_id) or (self.start_token_id == candidate.start_token_id and self.end_token_id == candidate.end_token_id)))

	# check if this candidate position-wise completely overlaps the given candidate.
	def Contains(self, candidate):
		return (self.start_token_id <= candidate.start_token_id and self.end_token_id >= candidate.end_token_id)

	# check if the uris of thie candidate overlap with the uris of the given candidate.
	def Meaning_overlaps_with(self, candidate):
		return (len(set([u.uri for u in self.uris]).intersection(set([u.uri for u in candidate.uris]))) > 0)

	# output: represent the pos tags of this candidate.
	def POS_output(self):
		return '+'.join(self.pos)


class Entity_candidate_uri(object):
	def __init__(self, uri, parent):
		self.parent = parent
		self.uri = uri['uri']
		self.uri_id = uri['id']
		self.base_score = 0.0
		self.score = 0.0
		self.uri_score = 0.0
		
		self.entity_relation_score = 0.0
		self.show_relations_for_output = False
		self.is_candidate = False
		self.consider = False
		self.relations = {}
		self.entity_confidence = 0.0

	# get the sum of the relation weights of the relations this uri has.
	def Relation_weight_sum(self):
		weight = 0
		for relation_list in self.relations.itervalues():
			for relation in relation_list:
				weight += features.Features.Relation_weight(relation)

		return weight

	# get the uri relation score of this uri.
	def Related_uri_score_sum(self):
		return sum(self.parent.module.uri_id_map[uri_id].score * sum(features.Features.Relation_weight(relation) for relation in relations) for uri_id, relations in self.relations.items())

# start-of-service initialization.
def Initialize(config):
	Config.Init(config)
	entity_relations.Config.Init(config)
	entity_base_scoring.Config.Init(config)
	features.Config.Init(config)
	features.Features.Init()

	config_domain = config['domains'][0]

	if config_domain['label_cache']['use']:
		labels = entity_labels.Get_from_file(config_domain['label_cache']['file'])
	else:
		labels = entity_labels.Get_from_endpoint(Config.endpoint, Config.graph_iri, config_domain['label_query'])

		if config_domain['label_cache']['write']:
			entity_labels.Write_to_file(labels, config_domain['label_cache']['file'])

	exact_surface_matching.Config.Init(labels)

	# labels is pretty big; we do not need this anymore.
	del labels

	# if config_domain['pred_freq_cache']['use']:
	# 	pred_freq = predicate_freq.Get_from_file(config_domain['pred_freq_cache']['file'])
	# else:
	# 	pred_freq = predicate_freq.Get_from_endpoint(Config.endpoint, Config.graph_iri, config_domain['pred_freq_query'])

	# 	if config_domain['pred_freq_cache']['write']:
	# 		predicate_freq.Write_to_file(pred_freq, config_domain['pred_freq_cache']['file'])

	# features.Features.Import_relation_weights(pred_freq)
	# del pred_freq

# entity linking.
def Entity_linking(request):
	try:
		request_json = json.loads(request)

		if 'text' not in request_json:
			raise Exception('"text" key required.')

		if 'lower_bound' in request_json:
			lower_bound = request_json['lower_bound']
		else:
			lower_bound = 0.5

		print("morph analysis...")
		tokens = Config.morph_analysis.Parse_text(request_json['text'])

		print("finding entity candidates...")
		matches = exact_surface_matching.Find_entity_candidates(tokens, overlapping=True)
		
		print("retrieving uri base scores...")
		entity_base_scores = entity_base_scoring.Entity_base_scores(matches)
		
		# print "finding uri relations..."
		# relations = entity_relations.Get_relations(tokens, matches)

		print("initiating entity linking...")
		entity_detection = Entity_detection(tokens, matches, entity_base_scores, lower_bound=lower_bound)
		
		# print "performing entity detection..."
		# entity_detection = Entity_detection(tokens, matches, entity_base_scores, relations)

		result = output.Output(entity_detection)

		return result.Response_repr()

	except Exception as e:
		print(traceback.format_exc(e))
		return json.dumps({"error":"error"})

# def Baseline_entity_linking(request):
# 	try:
# 		request_json = json.loads(request)

# 		if 'text' not in request_json:
# 			raise Exception('"text" key required.')

# 		print "morph analysis..."
# 		tokens = Config.morph_analysis.Parse_text(request_json['text'])

# 		print "finding entity candidates..."
# 		matches = exact_surface_matching.Find_entity_candidates(tokens, overlapping=True)
		
# 		print "retrieving uri base scores..."
# 		entity_base_scores = entity_base_scoring.Entity_base_scores(matches)
		
# 		# print "finding uri relations..."
# 		# relations = entity_relations.Get_relations(tokens, matches)

# 		print "initiating entity linking..."
# 		entity_detection = Baseline_entity_detection(tokens, matches, entity_base_scores)
		
# 		# print "performing entity detection..."
# 		# entity_detection = Entity_detection(tokens, matches, entity_base_scores, relations)

# 		result = output.Output(entity_detection)

# 		return result.Response_repr()

# 	except Exception as e:
# 		print traceback.format_exc(e)
# 		return json.dumps({"error":"error"})

# training.
def Train(request):
	try:
		request_json = json.loads(request)

		if 'text' not in request_json:
			raise Exception('"text" key required.')

		print("morph analysis...")
		tokens = Config.morph_analysis.Parse_text(request_json['text'])

		print("finding entity candidates...")
		matches = exact_surface_matching.Find_entity_candidates(tokens, overlapping=False)
		
		print("retrieving uri base scores...")
		entity_base_scores = entity_base_scoring.Entity_base_scores(matches)
		
		# print "finding uri relations..."
		# relations = entity_relations.Get_relations(tokens, matches)

		print("initiating learning process...")
		entity_detection_learning = Entity_detection_learning(tokens, matches, entity_base_scores)
		
		# print "performing entity detection..."
		# entity_detection = Entity_detection(tokens, matches, entity_base_scores, relations)

		return json.dumps({'hit':entity_detection_learning.comments['hit_requests'], 'total':entity_detection_learning.comments['total_requests']})

	except Exception as e:
		print(traceback.format_exc(e))
		return json.dumps({"error":"error"})

# def Parse_text_for_request(request):
# 	try:
# 		request_json = json.loads(request)

# 		if 'text' not in request_json:
# 			raise Exception('"text" key required.')

# 		return Config.morph_analysis.Parse_text_for_request(request_json['text'])

# 	except Exception as e:
# 		print traceback.format_exc(e)
# 		return json.dumps({"error":"error"})

# get the current state of trained features.
def Export_current_features():
	try:
		return json.dumps(features.Features.Export())

	except Exception as e:
		print(traceback.format_exc(e))
		return json.dumps({"error":"error"})


if __name__ == "__main__":
	sys.exit("use routing.py")
