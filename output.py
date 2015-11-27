# translates entity detection results into output format.
class Output(object):
	def __init__(self, entity_detection):
		self.actual_tokens = []
		self.actual_entities = []
		self.actual_relations = []
		self.comments = entity_detection.comments

		actual_token_id = 0
		last_entity_end = -1

		candidates = entity_detection.Candidates_for_output()

		#for candidate in candidates:
                        #if candidate.selected_uri is not None:
		# make a set of the existing uri ids.
		# some uris in relations might not exist due to overlapping issues.
		existing_uri_ids = set()
		for candidate in candidates:
			for uri in candidate.uris:
				existing_uri_ids.add(uri.uri_id)

		for candidate in candidates:

			# add all tokens before the entity.
			#print(candidate.text)
			for token in entity_detection.tokens[last_entity_end+1:candidate.start_token_id]:
				self.actual_tokens.append(Output_token(token, actual_token_id))
				actual_token_id += 1

			
			# add a new token representing the entity.
			entity_token = Output_token(entity_detection.tokens[candidate.start_token_id], actual_token_id)

			for append_token in entity_detection.tokens[candidate.start_token_id + 1 : candidate.end_token_id + 1]:
				entity_token.Add_token(append_token)

			entity_start_offset = entity_detection.tokens[candidate.start_token_id].start_offset
			entity_end_offset = entity_detection.tokens[candidate.end_token_id].end_offset

			self.actual_tokens.append(entity_token)

			# append the entity, based on the created token.
			self.actual_entities.append(Output_entity(candidate, actual_token_id, entity_start_offset, entity_end_offset))

			actual_token_id += 1
			last_entity_end = candidate.end_token_id

			# add relations starting from this entity, going right.
			for candidate_uri in candidate.uris:
				if candidate_uri.is_candidate:
					for related_uri_id, pred_list in candidate_uri.relations.items():
						if related_uri_id > candidate_uri.uri_id and related_uri_id in existing_uri_ids:
							self.actual_relations.append(Output_relation(candidate_uri.uri_id, related_uri_id, pred_list))

		# add all tokens after the last entity.
		for token in entity_detection.tokens[last_entity_end + 1:]:
			self.actual_tokens.append(Output_token(token, actual_token_id))
			actual_token_id += 1

		


	def Response_repr(self):
		return {
			'tokens': [token.Response_repr() for token in self.actual_tokens],
			'entities': [entity.Response_repr() for entity in self.actual_entities],
			'relations': [relation.Response_repr() for relation in self.actual_relations],
			'comments': self.comments
		}

class Output_token(object):
	def __init__(self, token, token_id):
		self.start_offset = token.start_offset,
		self.end_offset = token.end_offset,
		self.text = token.text
		self.pos = token.pos[:] # we want a deep copy of the list.
		self.sentence_id = token.sentence_id
		self.token_id = token_id
		self.is_word_start = token.is_word_start
		self.is_word_end = token.is_word_end

	def Add_token(self, token):
		if token.is_word_start:
			self.text += ' '

		self.text += token.text
		self.pos += token.pos
		self.is_word_end = token.is_word_end

	def Response_repr(self):
		return {
			'start_offset': self.start_offset,
			'end_offset': self.end_offset,
			'text': self.text,
			'pos': self.pos,
			'sentence_id': self.sentence_id,
			'token_id': self.token_id,
			'is_word_start': self.is_word_start,
			'is_word_end': self.is_word_end
		}

class Output_entity(object):
	def __init__(self, entity, token_id, start_offset, end_offset):
		self.text = entity.text
		self.token_id = token_id
		self.pos = entity.POS_output()
		self.is_entity = entity.is_entity or entity.chosen_for_learning
		if entity.selected_uri is not None:
			self.selected_uri_id = entity.selected_uri.uri_id
			self.selected_uri = entity.selected_uri.uri
			self.score = entity.selected_uri.score
		else:
			self.selected_uri_id = None
			self.selected_uri = None
			self.score = None
		self.uris = [Output_URI(uri) for uri in entity.uris]
		self.start_offset = start_offset
		self.end_offset = end_offset
		print(self.text,' ',self.is_entity,' ',self.score,' ',self.selected_uri,' ', self.pos)

	def Response_repr(self):
		return {
			'text': self.text,
			'token_id': self.token_id,
			'pos': self.pos,
			'uris': [uri.Response_repr() for uri in self.uris],
			'is_entity': self.is_entity,
			'selected_uri_id': self.selected_uri_id,
			'selected_uri': self.selected_uri,
			'score': self.score,
			'start_offset': self.start_offset,
			'end_offset': self.end_offset
		}

class Output_relation(object):
	def __init__(self, from_uri_id, to_uri_id, pred_list):
		self.from_uri_id = from_uri_id
		self.to_uri_id = to_uri_id
		self.pred_list = pred_list

	def Response_repr(self):
		return {
			'from': self.from_uri_id,
			'to': self.to_uri_id,
			'preds': self.pred_list
		}

class Output_URI(object):
	def __init__(self, uri):
		self.uri = uri.uri
		self.id = uri.uri_id
		self.score = uri.score
		self.uri_score = uri.uri_score
		self.entity_relation_score = uri.entity_relation_score
		self.entity_confidence = uri.entity_confidence
		#print(self.id,' --> ',self.uri,' score: ',round(self.score, 3),' uri_score: ',round(self.uri_score, 3),' entity_relation_score: ',round(self.entity_relation_score, 3),' entity_confidence: ',round(self.entity_confidence, 3))

	def Response_repr(self):
		return {
			'uri': self.uri,
			'id': self.id,
			'score': round(self.score, 3),
			'uri_score': round(self.uri_score, 3),
			'entity_relation_score': round(self.entity_relation_score, 3),
			'entity_confidence': round(self.entity_confidence, 3)
		}
