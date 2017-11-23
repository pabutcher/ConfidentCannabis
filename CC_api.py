# Import libraries
import os.path
from bottle import route, run, response, static_file, request, error, Bottle, template
from json import dumps, loads, load
from jsonschema import validate
import pandas as pd
import numpy as np
from CC_price_model import whlsle_prc

class CC_API:
    def __init__(self, port, local):
        self._app = Bottle()
        self._route() #Launch route
        self._local = local
        self._port = port
		# will want to load the model from pickle file. Or better to load in prediction function or separate function?
        if local:
            self._host = '127.0.0.1'
        else:
            self._host = '0.0.0.0'
		# Set default dict structure and types
        self.schema={"type" : "object", "properties" : {
            "date" : {"type" : "string"},
		    "date_sessiontime": {"type" : "number"},
		    "month": {"type" : "number"},
		    "retailer": {"type" : "string"},
		    "retailer_city": {"type" : "string"},
		    "processor": {"type" : "string"},
		    "processor_city": {"type" : "string"},
		    "producer": {"type" : "string"},
		    "producer_city": {"type" : "string"},
		    "retailer_to_seattle": {"type" : "number"},
		    "retailer_to_tacoma": {"type" : "number"},
		    "retailer_to_bellingham": {"type" : "number"},
		    "retailer_to_olympia": {"type" : "number"},
		    "retailer_to_bellingham": {"type" : "number"},
		    "retailer_to_vancouver": {"type" : "number"},
		    "retailer_to_kennewick": {"type" : "number"},
		    "retailer_to_yakima": {"type" : "number"},
		    "retailer_to_spokane": {"type" : "number"},
		    "strain_display_name": {"type" : "string"},
		    "product_name": {"type" : "string"},
		    "product_size_g": {"type" : "number"},
		    "distance": {"type" : "number"},
		    "total_lb_sold": {"type" : "number"},
		    "units_sold": {"type" : "number"},
		    "lab_name": {"type" : "string"},
		    "thc": {"type" : "number"},
		    "cbd": {"type" : "number"},
		    "moisture": {"type" : "number"},
		    "comp_1": {"type" : "number"},
		    "comp_2": {"type" : "number"},
		    "comp_3": {"type" : "number"},
		    "comp_4": {"type" : "number"},
            },
        }

    def start(self):
        self._app.run(server='paste', host=self._host, port=self._port)

    def _route(self):
        self._app.hook('before_request')(self._strip_path) # Needed to prevent errors.
        self._app.route('/', callback=self._homepage) # We tell to the API to listen on "/" and execute the action "_homepage()" when "/" is called

        # Response to Post
        self._app.route('/generate_wholesale_price', method="POST", callback=self._doAction)

        # Response to Get. Return error.
        self._app.route('/generate_wholesale_price', method="GET", callback=self._noinputerror)

    def _strip_path(self):
        request.environ['PATH_INFO'] = request.environ['PATH_INFO'].rstrip('/')

    def _homepage(self):
        return static_file("index.html", root=os.path.join(os.getcwd(),'html')) # Setup homepage

    def _noinputerror(self):
        rv = {"success": False,"payload": {"pplb_pretax":"error no json input"}}
        return dumps(rv)
		
    def _doAction(self):
        try:
            insamp_info = request.json['sample_info']
            valcheck=validate(insamp_info,self.schema) #validate schema
        except:
            rv = {"success": False,"payload": {"pplb_pretax":"incomplete json request"}}
            return dumps(rv)
        response.content_type = 'application/json'
        outsam_pplb = whlsle_prc(insamp_info) #model goes here
        rv = {"success": True,"payload": {"pplb_pretax":outsam_pplb}}
        return dumps(rv) # We dump the dictionary into json file and return it.