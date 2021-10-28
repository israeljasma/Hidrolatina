from tkinter import *
from PIL import Image, ImageTk
from Services import API_Services

from tkinter import messagebox

login = Tk()
login.geometry('350x500+500+50')
# login.overrideredirect(True)
login.resizable(0,0)
login.config(bg='#CCEEFF')
login.title('Hidrolatina')

#Def
def verification():
    user = userEntry.get()
    password = passwordEntry.get()
    person = API_Services.login(user, password)
    if 'token' in person:
        messagebox.showinfo(message=[person['user']['name'], person['user']['last_name']], title="Login")
    else:
        messagebox.showinfo(message=person['error'], title="Login")

def closeLogin():
    login.destroy()
    login.quit()

logo = Image.open('images/logo_hidrolatina.png')
logo = logo.resize((325, 97), Image.ANTIALIAS)
logo = ImageTk.PhotoImage(logo)
logoLabel = Label(login, image=logo, width=325, height=97, bg='#CCEEFF')
logoLabel.pack(pady=30)

userLabel = Label(login, text='Usuario', bg='#CCEEFF').pack()
userEntry = Entry()
userEntry.pack()

passwordLabel = Label(login, text='Contraseña', bg='#CCEEFF').pack()
passwordEntry = Entry(show='*')
passwordEntry.pack()

loginButton = Button(login, command=verification, text='Iniciar Sesión', bg='#c2eaff').pack()

closeButton = Button(login, text='Salir', command=closeLogin, bg='#c2eaff').pack()
login.mainloop()

# import tkinter as tk
# from tkinter import ttk
# import time


# class DigitalClock(tk.Tk):
#     def __init__(self):
#         super().__init__()

#         # configure the root window
#         self.title('Digital Clock')
#         self.resizable(0, 0)
#         self.geometry('250x80')
#         self['bg'] = 'black'

#         # change the background color to black
#         self.style = ttk.Style(self)
#         self.style.configure(
#             'TLabel',
#             background='black',
#             foreground='red')

#         # label
#         self.label = ttk.Label(
#             self,
#             text=self.time_string(),
#             font=('Digital-7', 40))

#         self.label.pack(expand=True)

#         # schedule an update every 1 second
#         self.label.after(1000, self.update)

#     def time_string(self):
#         return time.strftime('%H:%M:%S')

#     def update(self):
#         """ update the label every 1 second """

#         self.label.configure(text=self.time_string())

#         # schedule another timer
#         self.label.after(1000, self.update)


# if __name__ == "__main__":
#     clock = DigitalClock()
#     clock.mainloop()

# import tkinter as tk
# from tkinter import ttk
# import time


# class App(tk.Tk):
#     def __init__(self):
#         super().__init__()

#         self.title('Tkinter after() Demo')
#         self.geometry('300x100')

#         self.style = ttk.Style(self)

#         self.button = ttk.Button(self, text='Wait 3 seconds')
#         self.button['command'] = self.start
#         self.button.pack(expand=True, ipadx=10, ipady=5)

#     def start(self):
#         self.change_button_color('red')
#         self.change_button_color('black')

#     def change_button_color(self, color):
#         self.style.configure('TButton', foreground=color)


# if __name__ == "__main__":
#     app = App()
#     app.mainloop()