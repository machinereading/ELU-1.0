ELU : Entity Linking trained by Unannotated text

--------
Start-up
--------
python routing.py {CONFIG_FILE}

------
Config
------
{
	"host":{str} <IP of the service>
	"port":{str} <PORT # of the service>
	"lang":{str} <lang+tagger configuration (ko-etri, ja-mecab)>
	"domains":[{
		"name":{str} <appropriate name of the domain, not used>
		"endpoint":{str} <url of the SPARQL endpoint>
		"graph_iri":{str} <graph iri to use for querying, leave blank if N/A>
		"label_query":{str} <path to label SPARQL query txt file>
		"pred_freq_query":{str} <path to predicate frequency SPARQL query txt file>
		"relation_query":{str} <path to entity relations SPARQL query txt file>
		"subject_score_query":{str} <path to entity subject score SPARQL query txt file>
		"object_score_query":{str} <path to entity object score SPARQL query txt file>
		"priority":{int} <0, not used>
		"debug":{bool} <not used>
		"label_cache": {
			"file":{str} <path to the label cache file>
			"use":{bool} <whether to use to cache file on initialization>
			"write"{bool} <whether to write labels to the given file>
		}
		"pred_freq_cache": {
			"file":{str} <path to the predicate frequency cache file>
			"use":{bool} <whether to use to cache file on initialization>
			"write"{bool} <whether to write predicate frequencies to the given file>
		}
	}]
	"feature_weights": {
		"prev_pos":{float}
		"next_pos":{float}
		"start_pos":{float}
		"end_pos":{float}
		"has_pos":{float}
		"word_length":{float}
		"uri_object":{float}
		"relation_afterscore":{float}	
	}
	"recalculate_scores":{bool} <whether to recalculate entity scores using "feature_weights" on entity linking process>
	"use_svm":{bool} <whether to use SVM for training & entity linking>
	"features":{str} <path to the feature file>
	"ignore_numbers":{bool} {whether to ignore entity candidates consisting only of numbers}
}

--------------
Entity Linking
--------------

Request headers
---------------
Content-type:application/json

Request Body
------------
{
	"text":{str} text
	"lower_bound"{float} lower threshold of entity linking (0 <= x <= 1)
}

Response headers
----------------
Content-type:application/json

Response body
-------------
{
	"tokens":[{
		"sentence_id":{int}
		"token_id":{int}
		"text":{unicode}
		"start_offset":[{int}]
		"end_offset":[{int}]
		"pos":[{str},{str},...]
		"is_word_start":{bool}
	},...]
	"entities":[{
		"selected_uri":{text}
		"selected_uri_id":{int}
		"score":{float}
		"token_id":{int}
		"text":{text}
		"start_offset":{int}
		"end_offset":{int}
		"pos":{str}
		"is_entity":{bool}
		"uris":[{
			"entity_relation_score":{float}
			"uri":{str}
			"score":{float}
			"entity_confidence":{float}
			"uri_score":{float}
			"id":{int}
		},...]
	},...]
	"relations":[{
		"from":{int} <uri id>
		"to":{int} <uri id>
		"preds":[{str},{str},...]
	},...]
	"comments":{?} <extra comments>
}

--------
Training
--------

Request headers
---------------
Content-type:application/json

Request body
------------
{
	"text":{str}
}

Response headers
----------------
Content-type:application/json

Response body
-------------
(on success) <temporary response>
{
	"hit":{int} <relation cache hit #>
	"total":{int} <relation request #>
}

(on error)
{
	"error":"error"
}