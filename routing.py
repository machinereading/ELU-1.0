from bottle import post, run, route, response, request
import json
import urllib, urllib2
import sys

# custom imports
import entity_linking
# import morph_analysis

# the decorator
def enable_cors(fn):
	def _enable_cors(*args, **kwargs):
		# set CORS headers
		response.headers['Access-Control-Allow-Origin'] = '*'
		response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
		response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

		if request.method != 'OPTIONS':
			# actual request; reply with the actual response
			return fn(*args, **kwargs)

	return _enable_cors


# @post('/parse_text', method=['OPTIONS', 'POST'])
# @enable_cors
# def do_request():
# 	if not request.content_type.startswith('application/json'):
# 		return 'Content-type:application/json is required.'

# 	request_str = request.body.read()
# 	result_str = entity_linking.Parse_text_for_request(request_str)
	
# 	response.set_header('Content-type', 'application/json')
# 	return result_str


@post('/entity_linking', method=['OPTIONS', 'POST'])
@enable_cors
def do_request():
	if not request.content_type.startswith('application/json'):
		return 'Content-type:application/json is required.'

	request_str = request.body.read()
	result_str = entity_linking.Entity_linking(request_str)
	
	response.set_header('Content-type', 'application/json')
	return result_str

# @post('/baseline_entity_linking', method=['OPTIONS', 'POST'])
# @enable_cors
# def do_request():
# 	if not request.content_type.startswith('application/json'):
# 		return 'Content-type:application/json is required.'

# 	request_str = request.body.read()
# 	result_str = entity_linking.Baseline_entity_linking(request_str)
	
# 	response.set_header('Content-type', 'application/json')
# 	return result_str

@post('/training', method=['OPTIONS', 'POST'])
@enable_cors
def do_request():
	if not request.content_type.startswith('application/json'):
		return 'Content-type:application/json is required.'

	request_str = request.body.read()
	result_str = entity_linking.Train(request_str)
	
	response.set_header('Content-type', 'application/json')
	return result_str

@post('/features', method=['OPTIONS', 'GET'])
@enable_cors
def do_request():
	result_str = entity_linking.Export_current_features()
	
	response.set_header('Content-type', 'application/json')
	return result_str

@route('/')
def test():
	return 'ELU : Entity Linking trained by Unannotated text'

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: [python routing.py] [CONFIG_PATH]")

	with open(sys.argv[1]) as h:
		config = json.load(h)

	if 'host' not in config or 'port' not in config:
		sys.exit("config JSON requires 'host' and 'port' keys")

	entity_linking.Initialize(config)

	# result = entity_linking.Initialize(sys.argv[1])
	# if result == True:
	# 	print "entity_linking: Initialization complete."
	# 	run(host=config['host'], port=int(config['port']), debug=True)

	# else:
	# 	print result

	print "Initialization complete."
	run(host=config['host'], port=int(config['port']), debug=True)