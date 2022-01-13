import requests

url = 'http://127.0.0.1:8000/'
class API_Services():
    
    #Login
    def login(username, password):
        credentials = {'username':username,'password':password}
        request = requests.post(url + 'login/',data = credentials)
        request_dictionary = request.json()
        return request_dictionary

    def loginNFC(nfc):
        credentials = {'nfc':nfc}
        request = requests.post(url + 'nfclogin/', data = credentials)
        request_dictionary = request.json()
        return request_dictionary

    def logout(token, refresh):
        credentials = {'refresh': refresh}
        request = requests.post(url + 'logout/', data = credentials, headers={'Authorization': 'Bearer ' + token})
        request_dictionary = request.json()
        return request_dictionary

    #Users
    def userCreate(username, password, email, name, last_name, token, refresh=None):
        DATA  = {'password': password, 'username': username, 'email': email, 'name': name, 'last_name': last_name}
        request = requests.post(url + 'users/', json = DATA, headers={'Authorization': 'Bearer ' + token})
        request_dictionary = request.json()
        return request_dictionary

    def userList(token, refresh=None):
        credentials = {'refresh': refresh}
        request = requests.get(url + 'users/', data = credentials, headers={'Authorization': 'Bearer ' + token})
        request_dictionary = request.json()
        return request_dictionary

    def userRetrieve(id, token, refresh=None):
        credentials = {'refresh': refresh}
        request = requests.get(url + 'users/' + str(id) + '/', data = credentials, headers={'Authorization': 'Bearer ' + token})
        request_dictionary = request.json()
        return request_dictionary

    def userUpdate(id, username, email, name, last_name, token, refresh=None):
        DATA  = {'id': id, 'username': username, 'email': email, 'name': name, 'last_name': last_name}
        request = requests.put(url + 'users/' + str(id) + '/', data = DATA, headers={'Authorization': 'Bearer ' + token})
        request_dictionary = request.json()
        return request_dictionary

    def userDelete(id, token, refresh=None):
        DATA  = {'id': id}
        request = requests.delete(url + 'users/' + str(id) + '/', data = DATA, headers={'Authorization': 'Bearer ' + token})
        request_dictionary = request.json()
        return request_dictionary

    #PpeDetectin
    def ppeDetection(helmet, headphones, goggles, gloves, boots, token, refresh = None):
        credentials = {'helmet':helmet, 'headphones':headphones, 'goggles':goggles, 'gloves':gloves, 'boots':boots}
        request = requests.post(url + 'ppes/ppe/',data = credentials, headers={'Authorization': 'Bearer ' + token})
        request_dictionary = request.json()
        return request_dictionary

    #NFC
    def nfcCreate(nfc, active, token, refresh=None):
        DATA  = {'NFC': nfc, 'active': active}
        request = requests.post(url + 'identifications/nfc/', json = DATA, headers={'Authorization': 'Bearer ' + token})
        request_dictionary = request.json()
        return request_dictionary

    def nfcList(token, refresh=None):
        credentials = {'refresh': refresh}
        request = requests.get(url + 'identifications/nfc/', data = credentials, headers={'Authorization': 'Bearer ' + token})
        request_dictionary = request.json()
        return request_dictionary

    def nfcRetrieve(id, token, refresh=None):
        DATA = {'refresh': refresh}
        request = requests.get(url + 'identifications/nfc/' + str(id) + '/', data = DATA, headers={'Authorization': 'Bearer ' + token})
        request_dictionary = request.json()
        return request_dictionary

    def nfcUpdate(id, nfc, active, token, refresh=None):
        DATA  = {'id': id, 'NFC': nfc, 'active': active}
        request = requests.put(url + 'identifications/nfc/' + str(id) + '/', data = DATA, headers={'Authorization': 'Bearer ' + token})
        request_dictionary = request.json()
        return request_dictionary

    def nfcDelete(id, token, refresh=None):
        DATA  = {'refresh': refresh}
        request = requests.delete(url + 'identifications/nfc/' + str(id) + '/', json = DATA, headers={'Authorization': 'Bearer ' + token})
        request_dictionary = request.json()
        return request_dictionary