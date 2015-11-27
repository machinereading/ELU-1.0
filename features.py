from __future__ import division
import math
import codecs
import json
import bisect
from collections import defaultdict

# svm imports
import numpy
from sklearn import svm, cross_validation, metrics, preprocessing
from sklearn.externals import joblib

# learning, saving & loading of features.
# the average value of each feature will be 0.

class Config:
	features_file = None
	use_svm = None
	recalculate_scores = None
	feature_weights = None

	@staticmethod
	def Init(config):
		if 'features' in config:
			Config.features_file = config['features']

		if config['use_svm'] and Config.features_file is None:
			return Exception('use_svm=True requires a feature file.')

		Config.use_svm = config['use_svm']
		Config.recalculate_scores = config['recalculate_scores']

		if Config.recalculate_scores:
			Config.feature_weights = config['feature_weights']


class Features:
	uri_count = 0
	doc_count = 0
	entity_count = 0
	nonentity_count = 0
	uri_object_sum = 0.0
	uri_object_max = 0.0
	uri_subject_freq_multiplier = 1.0
	uri_object_freq_multiplier = 1.0
	word_entity_ratio = 5.0
	relation_limit = 0.20

	pos = None
	word_length = None
	uri_object = None
	relation_weights = None
	afterscore = None
	entity_svm_features = None
	feature_relation_afterscore = None
	trained_svm_model = None
	b = None
	c = None

	@staticmethod
	def Init():
		if Config.use_svm:
			# use svm model with trained data.
			with codecs.open(Config.features_file, encoding='utf-8') as h:
				features_json = json.loads(h.read())
			Features.Import_svm(features_json)		

		elif Config.features_file is not None:
			# use confidence model with trained data.
			with codecs.open(Config.features_file, encoding='utf-8') as h:
				features_json = json.loads(h.read())
			Features.Import(features_json)	
		
		else:
			# use confidence model... with no data.
			Features._Init_with_no_import()

	@staticmethod
	def Learn_kb_features(tokens, candidates, all_base_scores):
		for candidate in candidates:
			for uri in candidate.uris:
				Features.uri_count += 1
				base_scores = Features._Object_freq_base_scores(uri, all_base_scores)
				Features.uri_object_sum += base_scores
				Features.uri_object_max = max(Features.uri_object_max, base_scores)

		Features._Update_uri_feature_multiplier()

	@staticmethod
	def Learn_relation_weights(candidates):
		for candidate in candidates:
			for relation_list in candidate.representive_uri.relations.itervalues():
				for relation in relation_list:
					Features.relation_weights.Add_link(relation)

	@staticmethod
	def Relation_weight(link):
		return Features.relation_weights.Link_weight(link)

	@staticmethod
	def Learn_entity_features(candidates):
		for candidate in candidates:
			Features.pos.Learn_from_candidate(candidate)
			Features.word_length.Add_instance(candidate)
			Features.uri_object.Add_instance(candidate.representive_uri)

			if candidate.chosen_for_learning:
				Features.entity_count += 1
			else:
				Features.nonentity_count += 1

	@staticmethod
	def Learn_entity_afterscores(candidates):
		for candidate in candidates:
			Features.afterscore.Add_candidate(candidate)

	# svm model: save features of entities / non-entities.
	@staticmethod
	def Save_entity_features(candidates):
		for candidate in candidates:
			Features.entity_svm_features.Add_candidate(candidate)

	@staticmethod
	def Entity_feature_list(candidate, uri=None):
		if uri is None:
			use_uri = candidate.representive_uri
		else:
			use_uri = uri

		feature_list = []
		feature_list += Features.pos.Features(candidate)
		feature_list += Features.word_length.Feature(candidate)
		feature_list += Features.uri_object.Feature(use_uri)
		feature_list += Features.feature_relation_afterscore.Feature(use_uri)

		return feature_list


	@staticmethod
	def Learn_POS_features(candidates):
		for candidate in candidates:
			Features.pos.Learn_from_candidate(candidate)

		Features.pos.Adjust_scores()

	@staticmethod
	def Learn_word_length_features(candidates):
		for candidate in candidates:
			Features.word_length.Add_instance(candidate)

		Features.word_length.Adjust_multiplier()

	@staticmethod
	def Seed_entity_count(word_count):
		return int(math.ceil(word_count / Features.word_entity_ratio))

	@staticmethod
	def Seed_relation_ratio():
		return Features.relation_limit

	@staticmethod
	def Final_seed_entity_ratio_limit(given_entity_ratio):
		return (Features.word_entity_ratio * Features.doc_count + given_entity_ratio) / (Features.doc_count + 1)

	@staticmethod
	def Final_seed_relation_ratio_limit(given_relation_ratio):
		return (Features.relation_limit * Features.doc_count + given_relation_ratio) / (Features.doc_count + 1)

	@staticmethod
	def Update_ratio_limits(given_entity_ratio, given_relation_ratio):
		Features.word_entity_ratio = Features.Final_seed_entity_ratio_limit(given_entity_ratio)
		Features.relation_limit = Features.Final_seed_relation_ratio_limit(given_relation_ratio)
		Features.doc_count += 1


	@staticmethod
	def URI_initial_score(candidate, uri):
		return Features._Object_freq_score(uri)

	@staticmethod
	def URI_score_without_pos(candidate, uri, base_scores):
		return Features._Word_length_score(candidate) + Features._Object_freq_score(uri, base_scores)

	@staticmethod
	def URI_score(candidate, uri):
		if Config.use_svm:
			# return a score > 1 if the classifier accepts this candidate,
			# and a score < 0 otherwise.
			feature_list = [Features.Translate_feature_list(Features.Entity_feature_list(candidate, uri=uri), features_only=True)]
			test_data = numpy.asarray(feature_list)
			test_data_features = test_data[:, :8]

			predict_labels = Features.trained_svm_model.predict(test_data_features)
                        
			if int(predict_labels[0]) == 1:
				return 2
			else:
				return -2

		else:
			# return a score x where 0 <= x <= 1, to be translated into confidence.
			return (Features._Word_length_score(candidate) + Features.pos.Candidate_score(candidate) + Features._URI_object_score(uri) + Features._URI_relation_afterscore(uri)) / (Features.pos.feature_count + 3)

	@staticmethod
	def Feature_default_score():
		return Features.entity_count / (Features.entity_count + Features.nonentity_count)

	@staticmethod
	def Feature_damping_head():
		return Features._Feature_damping_factor() * Features.entity_count

	@staticmethod
	def Feature_damping_body():
		return Features._Feature_damping_factor() * (Features.entity_count + Features.nonentity_count)

	@staticmethod
	def _Feature_damping_factor():
		return 1 / Features.doc_count

	@staticmethod
	def Export():
		return {
			'doc_count': Features.doc_count,
			'entity_count': Features.entity_count,
			'nonentity_count': Features.nonentity_count,
			'word_entity_ratio': Features.word_entity_ratio,
			'relation_limit': Features.relation_limit,
			'pos': Features.pos.Export(),
			'word_length': Features.word_length.Export(),
			'uri_object': Features.uri_object.Export(),
			'relation_weights': Features.relation_weights.Export(),
			'afterscore': Features.afterscore.Export(),
			'svm_features': Features.entity_svm_features.Export(),
			'feature_relation_afterscore': Features.feature_relation_afterscore.Export()
		}

	@staticmethod
	def Import(features_json):
		Features.doc_count = features_json['doc_count']
		Features.entity_count = features_json['entity_count']
		Features.nonentity_count = features_json['nonentity_count']
		Features.word_entity_ratio = features_json['word_entity_ratio']
		Features.relation_limit = features_json['relation_limit']

		Features.word_length = Feature_word_length(features=features_json['word_length'])
		Features.pos = POS_features(features=features_json['pos'])
		Features.uri_object = Feature_uri_object(features=features_json['uri_object'])
		Features.relation_weights = Relation_weights(features=features_json['relation_weights'])
		
		if Config.use_svm or Config.recalculate_scores:
			Features.entity_svm_features = Entity_svm_features(features=features_json['svm_features'])
		
		Features.feature_relation_afterscore = Feature_relation_afterscore(features=features_json['feature_relation_afterscore'])

		if Config.recalculate_scores:
			Features.afterscore = Entity_afterscore(feature_list=features_json['svm_features'], feature_weights=Config.feature_weights)
		else:
			Features.afterscore = Entity_afterscore(features=features_json['afterscore'])

	@staticmethod
	def Import_svm(features_json):
		print("Training SVM classifier...")

		# we first import all trained feature functions,
		# because we need to translate features into feature values.
		Features.Import(features_json)

		# translate features into feature values.
		train_data_list = []
		for feature_list in features_json['svm_features']['nonentity']:
			train_data_list.append(Features.Translate_feature_list(feature_list, is_entity=False))

		for feature_list in features_json['svm_features']['entity']:
			train_data_list.append(Features.Translate_feature_list(feature_list, is_entity=True))

		train_data = numpy.asarray(train_data_list)
		del train_data_list

		train_data_features = train_data[:, :8]
		train_data_labels = train_data[:, 8]
		Features.b=train_data_features
		Features.c=train_data_labels
		
		# create the classifier.
		svm_args = {
			'kernel': 'poly',
			'cache_size': 1000,
			'class_weight': 'auto'
		}

		Features.trained_svm_model = svm.SVC(**svm_args)
		Features.trained_svm_model.fit(train_data_features, train_data_labels)


	@staticmethod
	def Translate_feature_list(feature_list, features_only=False, is_entity=True):
		vectorized_features = []
		vectorized_features += Features.pos.Vectorize_features(feature_list[:5])
		vectorized_features.append(Features.word_length.Vectorize_feature(feature_list[5]))
		vectorized_features.append(Features.uri_object.Vectorize_feature(feature_list[6]))
		vectorized_features.append(Features.feature_relation_afterscore.Vectorize_feature(feature_list[7]))

		if features_only:
			# we do not append the feature corresponding to the answer class here.
			return vectorized_features

		answer_feature = 1.0 if is_entity else 0.0
		vectorized_features.append(answer_feature)
		return vectorized_features

	@staticmethod
	def _Init_with_no_import():
		Features.doc_count = 1
		Features.entity_count = 1
		Features.nonentity_count = 1
		Features.uri_object_freq_multiplier = 1.0
		Features.word_length_multiplier = 1.0
		Features.word_entity_ratio = 6.0
		Features.relation_limit = 0.04
		Features.pos = POS_features()
		Features.word_length = Feature_word_length()
		Features.uri_object = Feature_uri_object()
		Features.relation_weights = Relation_weights()
		Features.afterscore = Entity_afterscore()
		Features.entity_svm_features = Entity_svm_features()
		Features.feature_relation_afterscore = Feature_relation_afterscore()

	@staticmethod
	def Import_relation_weights(pred_freq):
		Features.relation_weights = Relation_weights(pred_freq)

	@staticmethod
	def _Word_length_base_score(candidate):
		return len(candidate.text)

	@staticmethod
	def _Update_uri_feature_multiplier():
		Features.uri_object_freq_multiplier = 1 / Features.uri_object_max

	@staticmethod
	def _Object_freq_base_scores(uri):
		relation_weight_sum = 0.0
		relation_counts = uri.parent.module.uri_base_scores.Scores_of_uri_id(uri.uri_id)

		for link, count in relation_counts.items():
			relation_weight_sum += Features.Relation_weight(link) * count

		return math.log(relation_weight_sum+1.1)

	@staticmethod
	def _Word_length_score(candidate):
		# word length * multiplier - 1
		#print(candidate.text, ' --> ',Features.word_length.Score(candidate))
		return Features.word_length.Score(candidate)

	@staticmethod
	def _URI_object_score(uri):
		#print(uri.uri, ' --> ',Features.uri_object.Score(uri))
		return Features.uri_object.Score(uri)

	@staticmethod
	def _URI_relation_afterscore(uri):
		#print(uri.uri, ' --> ',uri.Related_uri_score_sum(),' / ',uri.uri_score)
		return uri.Related_uri_score_sum() / uri.uri_score

	@staticmethod
	def _Object_freq_score(uri):
		base_scores = Features._Object_freq_base_scores(uri)
		return base_scores * Features.uri_object_freq_multiplier


class POS_features(object):
	def __init__(self, features=None):
		self.feature_count = 5
		
		if features is not None:
			self.prev_pos = POS_feature_prev(features=features['prev_pos'])
			self.next_pos = POS_feature_next(features=features['next_pos'])
			self.start_pos = POS_feature_start(features=features['start_pos'])
			self.end_pos = POS_feature_end(features=features['end_pos'])
			self.has_pos = POS_feature_has(features=features['has_pos'])
		else:
			self.prev_pos = POS_feature_prev()
			self.next_pos = POS_feature_next()
			self.start_pos = POS_feature_start()
			self.end_pos = POS_feature_end()
			self.has_pos = POS_feature_has()

	def Learn_from_candidate(self, candidate):
		self.prev_pos.Add_instance(candidate)
		self.next_pos.Add_instance(candidate)
		self.start_pos.Add_instance(candidate)
		self.end_pos.Add_instance(candidate)
		self.has_pos.Add_instance(candidate)

	def Features(self, candidate):
		feature_list = []
		feature_list += self.prev_pos.Feature(candidate)
		feature_list += self.next_pos.Feature(candidate)
		feature_list += self.start_pos.Feature(candidate)
		feature_list += self.end_pos.Feature(candidate)
		feature_list += self.has_pos.Feature(candidate)

		return feature_list

	def Vectorize_features(self, feature_list):
		vector_list = []
		vector_list.append(self.prev_pos.Vectorize_feature(feature_list[0]))
		vector_list.append(self.next_pos.Vectorize_feature(feature_list[1]))
		vector_list.append(self.start_pos.Vectorize_feature(feature_list[2]))
		vector_list.append(self.end_pos.Vectorize_feature(feature_list[3]))
		vector_list.append(self.has_pos.Vectorize_feature(feature_list[4]))

		return vector_list

	def Candidate_score(self, candidate):
		score = 0
		score += self.prev_pos.Score(candidate)
		score += self.next_pos.Score(candidate)
		score += self.start_pos.Score(candidate)
		score += self.end_pos.Score(candidate)
		score += self.has_pos.Score(candidate)

		#print(candidate.text, ' --> ', score)
		return score

	def Export(self):
		return {
			'prev_pos': self.prev_pos.Export(),
			'next_pos': self.next_pos.Export(),
			'start_pos': self.start_pos.Export(),
			'end_pos': self.end_pos.Export(),
			'has_pos': self.has_pos.Export()
		}

	def Output_repr(self):
		return {
			'prev_pos': self.prev_pos.Output_repr(),
			'next_pos': self.next_pos.Output_repr(),
			'start_pos': self.start_pos.Output_repr(),
			'end_pos': self.end_pos.Output_repr(),
			'has_pos': self.has_pos.Output_repr()
		}

	@staticmethod
	def _Feature_score(feature, key):
		if key in feature:
			return feature[key].Score()
		else:
			return Features.Feature_default_score()

class Feature(object):
	def __init__(self, features=None):
		self.instance = {}

		if features is not None:
			# self.max = features['max']
			# self.damping = features['damping']
			# self._Adjust_scaling()
			
			for tag, tag_data in features['tags'].items():
				self.instance[self._Tag(tag)] = Instance_counter(features=tag_data)
		else:
			pass
			# self.multiplier = 1.0
			# self.average = 0.0
			# self.max = 0.0
			# self.scaling = 1.0
			# self.damping = 10000
			
	def Vectorize_feature(self, feature_key):
		return self._Score_of_key(feature_key)

	def Feature(self, candidate):
		return self._Get_feature_keys(candidate)

	def Add_instance(self, candidate):
		for tag in self._Get_feature_keys(candidate):
			if tag not in self.instance:
				self.instance[tag] = Instance_counter()

			self.instance[tag].Add_instance(self._Candidate_weight(candidate))

	def Score(self, candidate):
		keys = self._Get_feature_keys(candidate)
		return sum(self._Score_of_key(key) for key in keys) / len(keys)

	def Export(self):
		return {
			# 'max': self.max,
			'tags': dict([(tag, e.Export()) for tag, e in self.instance.items()])
		}

	def _Tag(self, tag):
		return tag

	def _Score_of_key(self, key):
		
		if key in self.instance:
			return self.instance[key].Instance_score()
		else:
			return self._Default_score(key)

	def _Default_score(self, key):
		return Features.Feature_default_score()

	def Output_repr(self):
		return u' '.join([unicode(key) + u"(" + unicode(round(self._Score_of_key(key), 2)) + u")" for key, instance in sorted(self.instance.items(), key=lambda x:x[1].Instance_score(), reverse=True)])

	def _Candidate_weight(self, candidate):
		if candidate.chosen_for_learning:
			return 1
		else:
			return 0

	def _Get_feature_keys(self, candidate):
		return []

class Feature_word_length(Feature):
	def _Tag(self, tag):
		return int(tag)

	def _Default_score(self, key):
		if key > 0:
			return self._Score_of_key(key-1)
		else:
			return Features.Feature_default_score()

	def _Get_feature_keys(self, candidate):
		return [len(candidate.text)]

class Feature_uri_object(Feature):
	def _Tag(self, tag):
		return int(tag)

	def _Default_score(self, key):
		if key > 0:
			return self._Score_of_key(key-1)
		else:
			return Features.Feature_default_score()

	def _Get_feature_keys(self, candidate_uri):
		return [int(math.ceil(candidate_uri.uri_score / 0.5))]

	def _Candidate_weight(self, candidate_uri):
		if candidate_uri.parent.chosen_for_learning:
			return 1
		else:
			return 0

class Feature_relation_afterscore(Feature):
	def _Tag(self, tag):
		return int(tag)

	def _Default_score(self, key):
		if key > 0:
			return self._Score_of_key(key-1)
		else:
			return Features.Feature_default_score()

	def _Get_feature_keys(self, candidate_uri):
		return [int(math.ceil(candidate_uri.Related_uri_score_sum() / 0.2))]

	def _Candidate_weight(self, candidate_uri):
		if candidate_uri.parent.chosen_for_learning:
			return 1
		else:
			return 0

class POS_feature_prev(Feature):
	def _Get_feature_keys(self, candidate):
		return [candidate.prev_pos]

class POS_feature_next(Feature):
	def _Get_feature_keys(self, candidate):
		return [candidate.next_pos]

class POS_feature_start(Feature):
	def _Get_feature_keys(self, candidate):
		return [candidate.pos[0]]

class POS_feature_end(Feature):
	def _Get_feature_keys(self, candidate):
		return [candidate.pos[-1]]

class POS_feature_has(Feature):
	def _Get_feature_keys(self, candidate):
		return candidate.pos

	def Feature(self, candidate):
		return ['+'.join(self._Get_feature_keys(candidate))]

	def Vectorize_feature(self, feature_key):
		keys = feature_key.split('+')
		return sum(self._Score_of_key(key) for key in keys)/len(keys)


class Instance_counter(object):
	def __init__(self, features=None):
		if features is not None:
			self.instances = features['count']
			self.sum = features['sum']
		else:
			self.instances = 0
			self.sum = 0.0
			

	def Add_instance(self, score):
		self.instances += 1
		self.sum += score

	def Instance_score(self):
		#print((self.sum + Features.Feature_damping_head()) / (self.instances + Features.Feature_damping_body()))
		#print(self.sum)
		#print(self.instances)
		return (self.sum + Features.Feature_damping_head()) / (self.instances + Features.Feature_damping_body())
		# return self.sum

	def Export(self):
		return {
			'count': self.instances,
			'sum': self.sum
		}

class Relation_weights(object):
	def __init__(self, features=None):
		self.links = defaultdict(int)
		self.sum = 0
		if features is not None:
			for link, count in features['links'].items():
				self.links[link] = count

			self.sum = features['sum']

	def Add_link(self, link):
		if link not in self.links:
			self.links[link] = 0

		self.links[link] += 1
		self.sum += 1

	def Link_weight(self, link):
		if link not in self.links:
			return 1
		else:
			return (self.sum - self.links[link]) / self.sum

	def Export(self):
		return {
			'links': self.links,
			'sum': self.sum
		}

class Entity_afterscore(object):
	feature_weight_keys_order = ["prev_pos", "next_pos", "start_pos", "end_pos", "has_pos", "word_length", "uri_object", "relation_afterscore"]

	def __init__(self, features=None, feature_list=None, feature_weights=None):
		self.entity_scores = []
		self.nonentity_scores = []
		self.sorted_entity_scores = None
		self.sorted_nonentity_scores = None

		if features is not None:
			self.entity_scores = features['entity']
			self.nonentity_scores = features['nonentity']
		elif feature_list is not None:
			print("Recalculating entity scores...")
			for entity_features in feature_list['entity']:
				self.entity_scores.append(Entity_afterscore._Recalculated_score(Features.Translate_feature_list(entity_features, features_only=True), feature_weights))

			for nonentity_features in feature_list['nonentity']:
				self.nonentity_scores.append(Entity_afterscore._Recalculated_score(Features.Translate_feature_list(nonentity_features, features_only=True), feature_weights))

	def Add_candidate(self, candidate):
		self.sorted_entity_scores = None
		self.sorted_nonentity_scores = None

		if candidate.chosen_for_learning:
			f = self.entity_scores
		else:
			f = self.nonentity_scores

		f.append(Features.URI_score(candidate, candidate.representive_uri))
		if len(f) > 10000:
			f.pop(0)

	def Entity_confidence(self, uri):
		if self.sorted_entity_scores is None:
			self.sorted_entity_scores = sorted(self.entity_scores)
		if self.sorted_nonentity_scores is None:
			self.sorted_nonentity_scores = sorted(self.nonentity_scores)

		entity_percentile = bisect.bisect_left(self.sorted_entity_scores, uri.score) / len(self.sorted_entity_scores)
		
		
		nonentity_percentile = (len(self.sorted_nonentity_scores) - bisect.bisect_right(self.sorted_nonentity_scores, uri.score)) / len(self.sorted_nonentity_scores)

		#print(uri.uri,' --> ' ,(entity_percentile / (entity_percentile + nonentity_percentile)))

		return entity_percentile / (entity_percentile + nonentity_percentile)

	def Export(self):
		return {
			'entity': self.entity_scores,
			'nonentity': self.nonentity_scores
		}

	@staticmethod
	def _Recalculated_score(feature_list, weights):
		return sum([feature_list[index] * weights[key] for index, key in enumerate(Entity_afterscore.feature_weight_keys_order)]) / len(Entity_afterscore.feature_weight_keys_order)

class Entity_svm_features(object):
	def __init__(self, features=None):
		self.entity_features = []
		self.nonentity_features = []

		if features is not None:
			self.entity_features = features['entity']
			self.nonentity_features = features['nonentity']

	def Add_candidate(self, candidate):
		if candidate.chosen_for_learning:
			f = self.entity_features
		else:
			f = self.nonentity_features

		f.append(Features.Entity_feature_list(candidate))
		if len(f) > 10000:
			f.pop(0)

	def Export(self):
		return {
			'entity': self.entity_features,
			'nonentity': self.nonentity_features
		}
