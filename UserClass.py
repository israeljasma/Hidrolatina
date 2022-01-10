class Person:
    def __init__(self, username, name, last_name, email, token, refreshToken):
        
        self.username = username
        self.name = name
        self.last_name = last_name
        self.email = email
        self.token = token
        self.refreshToken = refreshToken

    def getUsername(self):
        return self.username

    def getName(self):
        return self.name

    def getLast_name(self):
        return self.last_name

    def getEmail(self):
        return self.email

    def getToken(self):
        return self.token

    def getRefreshToken(self):
        return self.refreshToken