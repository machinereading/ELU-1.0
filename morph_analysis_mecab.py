# -*- coding: utf-8 -*-
# Transforms MeCab-parsed text into a format to be used by ELU.
# !! Offsets are based on UNICODE: each character has a length of 1 !!

# period = 句点
# EOF might appear in the middle of parses, due to buffer overflows.
# we ignore these EOFs.

import json
import re
import subprocess
import codecs
# import traceback

# custom imports
import token_template

# testing: return parse results for the given text.
def Parse_text_for_request(text):
	parsed_tokens = Parse_text(text)
	return json.dumps(parsed_tokens.Response_repr())

# returns parse data (tailored for ELU) for the given text.
def Parse_text(text):
	mecab_lines = _Get_MeCab_parse(text)

	# prepare to create tokens for this text.
	tokens = token_template.Tokens()

	# get the POS parse of the text,
	# we ignore POS tokens.
	# we assume the first 2 'tags' (delimited by ,) to be the POS of a single morpheme.
	# we also assume that MeCab doesn't alter text. (this might be incorrect)
	current_sentence_id = 0
	current_offset_start = 0

	for line in mecab_lines:
		if line[:3] == "EOS":
			continue

		text, annotations = line.rstrip().split('\t')
		annotation_tokens = annotations.split(',')
		pos = annotation_tokens[0] + u',' + annotation_tokens[1]

		tokens.Add_token(
			current_offset_start,
			current_offset_start + len(text),
			text,
			[pos],
			current_sentence_id,
			False,
			False)

		tokens.Increment_word_id()

		current_offset_start += len(text)

		if annotation_tokens[1] == u'句点':
			current_sentence_id += 1

	return tokens

# gets the MeCab parse result for the given text.
# we use the file I/O interface of MeCab.
def _Get_MeCab_parse(text):
	in_file = 'mecab-temp.in'
	out_file = 'mecab-temp.out'

	with codecs.open(in_file, 'w', encoding='utf-8') as h:
		h.write(text)

	subprocess.call(['/usr/local/bin/mecab', in_file, '-o', out_file])

	results = []

	with codecs.open(out_file, encoding='utf-8') as h:
		for line in h:
			results.append(line)

	return results