# detects possible entity candidates within text, based on exact string matching.
# more specifically, we use tokens representing text.

from collections import defaultdict

class Config:
	label_tree = None

	@staticmethod
	def Init(labels):
		print("Initializing exact surface matching label tree.")
		Config.label_tree = Label_tree(labels)

# a character -> tree recursive tree structure.
# "abcd" -> ["a"]["b"]["c"]["d"]
class Label_tree(object):
	def __init__(self, labels):
		self.root = Label_tree_node()

		load_status = 0
		for surface, uris in labels.iteritems():
			load_status += 1
			if load_status % 10000 == 0:
				print(load_status), 

			self.root.Add_child(surface, uris)


class Label_tree_node(object):
	def __init__(self):
		self.children = {}
		self.uris = None

	def Add_child(self, remaining_text, uris):
		if len(remaining_text) > 0:
			if remaining_text[0] not in self.children:
				self.children[remaining_text[0]] = Label_tree_node()

			self.children[remaining_text[0]].Add_child(remaining_text[1:], uris)
		else:
			self.uris = uris

	def Traverse(self, remaining_text):
		# print remaining_text,
		if len(remaining_text) > 0:
			if remaining_text[0] not in self.children:
				# print 'X',
				return None
			else:
				# print '->',
				return self.children[remaining_text[0]].Traverse(remaining_text[1:])
		
		else:
			# print 'O',
			return self


class Label_tree_traverser(object):
	def __init__(self, start_token, start_token_id):
		self.start_token_id = start_token_id
		self.end_token_id = start_token_id
		
		self.node = Config.label_tree.root
		self.currently_matching_uris = None
		self.alive = True
		self.traverse_text = ""

		# start traversing with the token.
		self.Traverse_with_token(start_token, start_token_id, is_first=True)

	def Traverse_with_token(self, token, token_id, is_first=False):
		if not self.alive:
			pass

		self.end_token_id = token_id

		# traverse the tree with the given token.
		# if the token is the start of a word, whitespace must be before it.
		traverse_text = token.text
		if not is_first and token.is_word_start:
			traverse_text = u" " + token.text

		self.traverse_text += traverse_text

		self.node = self.node.Traverse(traverse_text)

		if self.node is None:
			# the traverse has ended.
			self.currently_matching_uris = None
			self.alive = False

		elif self.node.uris is not None:
			# the traverse is currently on a valid surface form.
			self.currently_matching_uris = self.node.uris

		else:
			# the traverse has not ended, but the current text is not a valid surface form.
			self.currently_matching_uris = None

	def Is_alive(self):
		return self.alive

	def Is_currently_valid(self):
		return (self.currently_matching_uris is not None)


class Matches(object):
	def __init__(self, tokens, candidates):
		self.matches = []
		self._uri_id = 0
		self._uri_id_to_match = []

		for candidate in candidates:
			self.matches.append(Match(tokens, candidate['start'], candidate['end'], self._URIs(candidate['uris'])))

	def _URIs(self, candidate_uris):
		uris = []
		for candidate_uri in candidate_uris:
			uris.append({
				'uri': candidate_uri,
				'id': self._uri_id})

			# make a uriID -> match link.
			self._uri_id_to_match.append(len(self.matches))

			self._uri_id += 1

		return uris

	def Match_of_uri_id(self, uri_id):
		return self.matches[self._uri_id_to_match[uri_id]]

	def __iter__(self):
		self.index = -1
		return self

	def __next__(self):
		if self.index == len(self.matches) - 1:
			raise StopIteration
		else:
			self.index += 1
			return self.matches[self.index]

	def __getitem__(self, i):
		return self.matches[i]


class Match(object):
	def __init__(self, tokens, start_token_id, end_token_id, uris):
		self.start = start_token_id
		self.end = end_token_id
		self.uris = uris


# returns: Matches
def Find_entity_candidates(tokens, overlapping=True):
	current_traversers = []
	candidates = []

	for token_id, token in enumerate(tokens):
		# print token.text

		# continue existing traversers with the current token.
		for traverser in current_traversers:
			traverser.Traverse_with_token(token, token_id)

		# add a new traverser with the current token.
		current_traversers.append(Label_tree_traverser(token, token_id))

		# remove traversers that are dead.
		current_traversers = list(filter(lambda t: t.Is_alive(), current_traversers))

		# get the token boundary and uris of traversers that are currently valid.
		for traverser in current_traversers:
			if traverser.Is_currently_valid():
				# print "(", traverser.start_token_id, traverser.end_token_id, "," + traverser.traverse_text + ")",
				# for uri in traverser.currently_matching_uris:
				# 	print uri,

				# print ''
				# print list(traverser.currently_matching_uris)[0]
				candidates.append({
					'start': traverser.start_token_id,
					'end': traverser.end_token_id,
					'uris': traverser.currently_matching_uris})

	if not overlapping:
		# we need to remove overlapping tokens.
		# we do this in left-to-right order, greedily taking the longest ones.
		current_end_threshold = -1
		independant_candidates = []

		for candidate in sorted(candidates, key=lambda c: (c['start'], c['end'] * -1)):
			if candidate['start'] > current_end_threshold:
				independant_candidates.append(candidate)
				current_end_threshold = candidate['end']

		return Matches(tokens, independant_candidates)

	else:
		return Matches(tokens, candidates)
