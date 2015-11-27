# Transforms ETRI-parsed text into a format to be used by ELU.
# !! Offsets are based on the UTF-8 ENCODED STRING: 1 Korean character usually has a length of 3 !!

import urllib
import urllib.request
from urllib.request import urlopen
import json
import re
import traceback

# custom imports
import token_template

# this portion is hard-coded because the current I/O specifications are not standardized.
# some way to refactor this part into configurable code?
class Config:
	service_url = 'http://143.248.135.187:22334/controller/service/etri_parser'
	headers = {
		'Content-type': 'application/x-www-form-urlencoded',
		'charset':'UTF-8'
	}

# testing: return parse results for the given text.
def Parse_text_for_request(text):
	parsed_tokens = Parse_text(text)
	return json.dumps(parsed_tokens.Response_repr())

# returns parse data (tailored for ELU) for the given text.
def Parse_text(text):
	etri_json = _Get_ETRI_parse(text)

	# prepare to create tokens for this text.
	tokens = token_template.Tokens()

	# get the POS parse of the text,
	# taking special care to tag blank spaces as separate tokens.
	# we assume that all sentences are separated by a blank space.
	for sentence_id, sentence_json in enumerate(etri_json['sentence']):
		morphs = sentence_json['morp']

		for word_index, word in enumerate(sentence_json['word']):
			word_tokens = Text_preserving_tokens(word['text'], morphs[word['begin']:word['end'] + 1])

			for token_index, word_token_data in enumerate(word_tokens):
				tokens.Add_token(
					word_token_data[0],
					word_token_data[1],
					word_token_data[2],
					word_token_data[3],
					sentence_id,
					(token_index == 0),
					(token_index == len(word_tokens) - 1))

			tokens.Increment_word_id()

	return tokens


# returns a list of morph token data for the given text & list of morphs.
# this list will have no tokens that have altered text.
def Text_preserving_tokens(text, morphs):
	tokens_data = []

	# we need to work in UTF-8 offsets.
	# if the encoding process is too slow, we need to manually translate offsets.
	utf8_text = text.encode('utf-8')

	start_offset = morphs[0]['position']

	current_queued_start_offset = None
	current_queued_pos_tags = []

	for morph in morphs:
		current_morph_utf8 = morph['lemma'].encode('utf-8')
		current_morph_start_offset = morph['position'] - start_offset

		if current_morph_utf8 == utf8_text[current_morph_start_offset:current_morph_start_offset + len(current_morph_utf8)]:
			# the current morph is unaltered!
			pos_tags = []

			# first push the current queued text if it exists.
			# ETRI is a bit inconsistent here; it should not matter too much. 
			if current_queued_start_offset is not None:
				pos_tags += current_queued_pos_tags				

			pos_tags.append(morph['type'])

			# push the current morph.
			tokens_data.append((current_morph_start_offset + start_offset, current_morph_start_offset + len(current_morph_utf8) + start_offset, current_morph_utf8.decode('utf-8'), pos_tags))

			# reset the queue.
			current_queued_start_offset = None
			current_queued_pos_tags = []

		else:
			# the current morph seems to be altered.
			# if this is the first altered morph, queue its start offset.
			if current_queued_start_offset is None:
				current_queued_start_offset = current_morph_start_offset

			# add the morph's POS tag to the queue.
			current_queued_pos_tags.append(morph['type'])

	# if any queued morphs remain, add them.
	if current_queued_start_offset is not None:
		tokens_data.append((current_queued_start_offset + start_offset, current_queued_start_offset + len(utf8_text[current_queued_start_offset:]) + start_offset, utf8_text[current_queued_start_offset:].decode('utf-8'), current_queued_pos_tags))

	return tokens_data


# gets the ETRI parse result for the given text.	
def _Get_ETRI_parse(text):
	request_data = urllib.parse.urlencode({'sentence': text})
	request_data=request_data.encode('utf-8')
	request = urllib.request.Request(Config.service_url, request_data, Config.headers)
	response = urlopen(request)
	response=response.read().decode('utf-8')
	return json.loads(response)
