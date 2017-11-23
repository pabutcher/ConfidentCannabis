# PAB 2017
# Import libraries
import os.path, sys
import xml.etree.ElementTree as XML
import CC_api #Import API from CC_api.py file

# Setup Server
def ServerConf(confPath = 'conf.xml'):
    configurations = XML.parse(confPath).getroot()
    servers = dict()
    for serv in configurations.iter('APIserver'):
        serverName = serv.attrib['serverName']
        serverPort = serv.attrib['port']
        serverIP = serv.attrib['ip']
        serverLocal = serv.attrib['local']
        servers[serverName] = {'ip':serverIP, 'port':serverPort, 'local':str2bool(serverLocal)}
    return servers

def str2bool(v):
	return v.lower() == "true"

# Run Server
configAPI = ServerConf() 

#API settings
serverAPI = {'port':configAPI['serverAPI']['port'], 'local':configAPI['serverAPI']['local']}

#Instanciate the api
api = CC_api.CC_API(serverAPI['port'], serverAPI['local'])

#Launch
api.start()