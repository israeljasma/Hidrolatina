from datetime import datetime, timedelta
from sys import path
from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox, filedialog, simpledialog, Listbox
import os
import platform
import time
from threading import Thread, Lock

from Services import API_Services
from UserClass import Person
from FileManagementClass import FileManagement
from NFCClass import NFC
from WindowsTk import WindowsTk

from numpy import CLIP
from imagenClipClass import imageClip




if __name__ == '__main__':
    root = Tk()
    # root.geometry('350x500+500+50')
    root.geometry(f'{root.winfo_screenwidth()}x{root.winfo_screenheight()}')
    # root.resizable(0,0)
    # root.overrideredirect(True)
    root.config(bg='#CCEEFF')
    root.title('Hidrolatina')
      

    instanceWindowsTk = WindowsTk(root)

    
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
            instanceWindowsTk.adminConfigTk(user)
        else:
            messagebox.showinfo(message=person['error'], title="Login")
        # except:
        #     messagebox.showerror(title='Error de conexión', message='No se ha podido establecer una conexión con el servidor. Comuníquese con su encargado de TI.')
        

    def closeLogin():
        try:
            instanceWindowsTk.p0.terminate()
            instanceWindowsTk.p0.join()
        except:
           pass
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
                instanceWindowsTk.nfc_identifyTk()
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
        NFC(instanceWindowsTk.nfc_identifyTk, instanceWindowsTk.showPytorchCameraTk)

    # Var
    startTime = datetime.now()
    endTime = datetime.now() + timedelta(seconds=120)
    boolCounter = True

    #Canvas
    canvas = Canvas(root, borderwidth=0,highlightthickness=0)
    canvas.place(relx=.5, rely=.5, relwidth=1, relheight=1, anchor='center')

    bg = Image.open('images/network_bg.png')
    bg = bg.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.ANTIALIAS)
    photoimage = ImageTk.PhotoImage(bg)
    canvas.create_image(root.winfo_screenwidth()/2, root.winfo_screenheight()/2, image=photoimage)


    # Labels and Buttons
    logo = Image.open('images/logo_hidrolatina.png')
    logo = logo.resize((325, 97), Image.ANTIALIAS)
    logo = ImageTk.PhotoImage(logo)
    # logoLabel = Canvas(root, borderwidth=0,highlightthickness=0, bg='#FFFFFF', width=logo.width(), height=logo.height())
    # logoLabel.place(relx=.5, rely=.1)
    print('tamaño: ', logo.width(), logo.height())
    canvas.create_image(root.winfo_screenwidth()/2, logo.height(), image=logo, anchor='center')
    # logoLabel = Label(root, image=logo, width=325, height=97, bg='#CCEEFF')
    # logoLabel.pack(pady=30)
    

    userLabel = Label(root, text='Usuario', bg='white').pack(pady=(200,0))
    userEntry = Entry(root)
    userEntry.bind("<1>", handle_click)
    userEntry.pack()

    passwordLabel = Label(root, text='Contraseña', bg='white').pack()
    passwordEntry = Entry(root, show='*')
    passwordEntry.pack()

    loginButton = Button(root, command=lambda:verification(), text='Iniciar Sesión', bg='#c2eaff').pack(pady=(50,0))
    # identificationButton = Button(root, command=lambda:nfc_identifyTk(), text='Iniciar Identificación', bg='#c2eaff').pack()
    identificationButton = Button(root, command=lambda:iniciarIdentificacionNFC(), text='Iniciar Identificación', bg='#c2eaff').pack()

    closeButton = Button(root, text='Salir', command=closeLogin, bg='#c2eaff').pack(pady=(50,0))

    def load():
        if messagebox.askyesno(message="Se cargarán dependecias \n ¿Desea continuar?"):
            instanceWindowsTk.loadALL() 
        root.after(2000,root.deiconify)
        instanceWindowsTk.center_window(root)
    

    root.withdraw()
    root.after(1, load)  
    # Call def
    root.after(10000, counter, endTime)


    root.mainloop()

    