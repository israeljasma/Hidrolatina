import requests

url = 'http://127.0.0.1:8000/'
class API_Services():
    
    def login(username="doravan", password="macdato13"):
        credentials = {'username':username,'password':password}
        request = requests.post(url + 'login/',data = credentials)
        request_dictionary = request.json()
        return request_dictionary

    def loginNFC(nfc):
        credentials = {'nfc':nfc}
        request = requests.post(url + 'nfclogin/',data = credentials)
        request_dictionary = request.json()
        return request_dictionary

    def ppeDetection(helmet, headphones, goggles, gloves, boots, token):
        credentials = {'helmet':helmet, 'headphones':headphones, 'goggles':goggles, 'gloves':gloves, 'boots':boots, 'token':token}
        request = requests.post('http://127.0.0.1:8000/ppe/ppe/',data = credentials)
        request_dictionary = request.json()
        return request_dictionary