class Person:
    def __init__(self, username, last_name, email, token):
        
        self.username = username
        self.last_name = last_name
        self.email = email
        self.token = token

    def getUsername(self):
        return self.username

    def getLast_name(self):
        return self.last_name

    def getEmail(self):
        return self.email

    def getToken(self):
        return self.token