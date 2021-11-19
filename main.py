from datetime import datetime, timedelta
from sys import path
from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox, filedialog, simpledialog, Listbox
import os
import platform
import time
from threading import Thread, Lock
from effdet.utils.inference import init_effdet_model,inference_effdet_model

from Services import API_Services
from UserClass import Person
from FileManagementClass import FileManagement
from NFCClass import NFC
from WindowsTk import WindowsTk

from numpy import CLIP
from imagenClipClass import imageClip


if __name__ == '__main__':
    root = Tk()
    root.geometry('350x500+500+50')
    root.resizable(0,0)
    root.config(bg='#CCEEFF')
    root.title('Hidrolatina')

    # Def
    def verification():
        user = userEntry.get()
        password = passwordEntry.get()
        # try:
        person = API_Services.login(user, password)
        if 'token' in person:
            user = Person(person['user']['username'], person['user']['name'], person['user']['last_name'], person['user']['email'], person['token'])
            #Hide Root Window
            # root.withdraw()
            # adminConfigTk(user)
            WindowsTk().adminConfigTk(user)
        else:
            messagebox.showinfo(message=person['error'], title="Login")
        # except:
        #     messagebox.showerror(title='Error de conexión', message='No se ha podido establecer una conexión con el servidor. Comuníquese con su encargado de TI.')
        

    def closeLogin():
        root.destroy()
        root.quit()

    def handle_click(event):
        print("clicked!")
        global boolCounter
        boolCounter = False

    def counter(endTime):
        if boolCounter:
            if datetime.now() > endTime:
                print('si')
                print(datetime.now().strftime('%H:%M:%S'), endTime.strftime('%H:%M:%S'))
                nfc_identifyTk()
            else:
                print('no')
                print(datetime.now().strftime('%H:%M:%S'), endTime.strftime('%H:%M:%S'))
                time.sleep(1)
                root.after(10000, counter, endTime)
        else:
            print('counter detenido')

        return

    def iniciarIdentificacionNFC():
        global boolCounter
        boolCounter = False
        NFC(nfc_identifyTk, showPytorchCameraTk)

    # Var
    startTime = datetime.now()
    endTime = datetime.now() + timedelta(seconds=120)
    boolCounter = True

    # Labels and Buttons
    logo = Image.open('images/logo_hidrolatina.png')
    logo = logo.resize((325, 97), Image.ANTIALIAS)
    logo = ImageTk.PhotoImage(logo)
    logoLabel = Label(root, image=logo, width=325, height=97, bg='#CCEEFF')
    logoLabel.pack(pady=30)

    userLabel = Label(root, text='Usuario', bg='#CCEEFF').pack()
    userEntry = Entry()
    userEntry.bind("<1>", handle_click)
    userEntry.pack()

    passwordLabel = Label(root, text='Contraseña', bg='#CCEEFF').pack()
    passwordEntry = Entry(show='*')
    passwordEntry.pack()

    loginButton = Button(root, command=lambda:verification(), text='Iniciar Sesión', bg='#c2eaff').pack()
    # identificationButton = Button(root, command=lambda:nfc_identifyTk(), text='Iniciar Identificación', bg='#c2eaff').pack()
    identificationButton = Button(root, command=lambda:iniciarIdentificacionNFC(), text='Iniciar Identificación', bg='#c2eaff').pack()

    closeButton = Button(root, text='Salir', command=closeLogin, bg='#c2eaff').pack()

    # Call def
    root.after(10000, counter, endTime)

    root.mainloop()