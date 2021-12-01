import requests

class API_Services():
    
    def login(username, password):
        credentials = {'username':username,'password':password}
        request = requests.post('http://127.0.0.1:8000/',data = credentials)
        request_dictionary = request.json()
        return request_dictionary

    def loginNFC(nfc):
        credentials = {'nfc':nfc}
        request = requests.post('http://127.0.0.1:8000/nfclogin/',data = credentials)
        request_dictionary = request.json()
        return request_dictionary

    def ppeDetection(helmet, headphones, goggles, gloves, boots, token):
        credentials = {'helmet':helmet, 'headphones':headphones, 'goggles':goggles, 'gloves':gloves, 'boots':boots, 'token':token}
        request = requests.post('http://127.0.0.1:8000/ppe/ppe/',data = credentials)
        request_dictionary = request.json()
        return request_dictionary