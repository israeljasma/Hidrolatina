import requests

class API_Services():
    
    def login(username, password):
        credentials = {'username':username,'password':password}
        request = requests.post('http://127.0.0.1:8000/',data = credentials)
        request_dictionary = request.json()
        return request_dictionary
