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

from numpy import CLIP, trunc
from imagenClipClass import imageClip

if __name__ == '__main__':
    root = Tk()
    # root.geometry('350x500+500+50')
    
    # root.resizable(0,0)
    root.overrideredirect(True)
    root.config(bg='#CCEEFF')
    root.title('Hidrolatina')
    root.geometry(f'{root.winfo_screenwidth()}x{root.winfo_screenheight()}')

      

    instanceWindowsTk = WindowsTk(root)

    
    # Def
    def verification():
        user = userEntry.get()
        password = passwordEntry.get()
        global boolCounter
        boolCounter = False
        # try:
        person = API_Services.login(user, password)
        if 'token' in person:
            user = Person(person['user']['username'], person['user']['name'], person['user']['last_name'], person['user']['email'], person['token'], person['refresh-token'])
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
        except:
           pass
        try:
            instanceWindowsTk.p0_1.terminate()
        except:
           pass
        try:
            instanceWindowsTk.p1.terminate()
        except:
           pass
        try:
            instanceWindowsTk.p2.terminate()
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
                # instanceWindowsTk.nfc_identifyTk()
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

    def on_focus_in(entry):
        if entry.cget('state') == 'disabled':
            entry.configure(state='normal')
            entry.delete(0, 'end')


    def on_focus_out(entry, placeholder):
        if entry.get() == "":
            entry.insert(0, placeholder)
            entry.configure(state='disabled')

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

    canvas.create_image(root.winfo_screenwidth()/2, logo.height(), image=logo, anchor='center')

    

    userLabel = Label(root, text='Rut', bg='white').pack(pady=(200,0))
    userEntry = Entry(root)
    userEntry.bind("<1>", handle_click)
    userEntry.pack()
    userEntry.insert(0, "11.111.111-1")
    userEntry.configure(state='disabled')

    passwordLabel = Label(root, text='Contraseña', bg='white').pack()
    passwordEntry = Entry(root, show='*')
    passwordEntry.pack()
    passwordEntry.insert(0, "Contraseña")
    passwordEntry.configure(state='disabled')

    x_focus_in = userEntry.bind('<Button-1>', lambda x: on_focus_in(userEntry))
    x_focus_out = userEntry.bind('<FocusOut>', lambda x: on_focus_out(userEntry, '11.111.111-1'))

    y_focus_in = passwordEntry.bind('<Button-1>', lambda x: on_focus_in(passwordEntry))
    y_focus_out = passwordEntry.bind('<FocusOut>', lambda x: on_focus_out(passwordEntry, 'Contraseña'))

    loginButton = Button(root, command=lambda:verification(), text='Iniciar Sesión', bg='#c2eaff').pack(pady=(50,0))
    # identificationButton = Button(root, command=lambda:nfc_identifyTk(), text='Iniciar Identificación', bg='#c2eaff').pack()
    identificationButton = Button(root, command=lambda:iniciarIdentificacionNFC(), text='Iniciar Identificación', bg='#c2eaff').pack()
    # popupButton = Button(root, command=lambda:instanceWindowsTk.popupIdentificationTk(booleanAnswerlist=[False, False, False, False, False, False]), text='POP UP', bg='#c2eaff').pack()
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
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes,'-topmost',False)
    root.mainloop()

    