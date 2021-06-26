from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox
from tkinter import filedialog
import sqlite3
import os
import platform

##Path
if platform.system() == "Darwin":
    print("MacOS")
elif platform.system() == "Linux":
    print("Linux")
elif platform.system() == "Windows":
    print("Windows")
    DATA_DIR = os.path.join('c:/hidrolatina', 'data')
    MODELS_DIR = os.path.join(DATA_DIR, 'models')
    dir = [DATA_DIR, MODELS_DIR]
    for dir in [DATA_DIR, MODELS_DIR]:
        if not os.path.exists(dir):
            os.makedirs(dir)


# if platform.system() == "Darwin":
#     print("MacOS")
# elif platform.system() == "Linux":
#     print("Linux")
# elif platform.system() == "Windows":
#     print("Windows")
#     DATA_DIR = os.path.join('c:\hidrolatina', 'data')
#     DATA_DIR = DATA_DIR.replace("\\","/")
#     print(DATA_DIR)
#     MODELS_DIR = os.path.join(DATA_DIR, 'models')
#     MODELS_DIR = MODELS_DIR.replace("\\","/")
#     print(MODELS_DIR)
#     for dir in [DATA_DIR, MODELS_DIR]:
#         if not os.path.exists(dir):
#             os.makedirs(dir)

##Data Base
#Crate conecction
#connection = sqlite3.connect('hidrolatina.db')

#Create Cursor
#c = connection.cursor()

#Create table
#    path text)""")
#c.execute("""CREATE TABLE mytable(

#Commit changes
#connection.commit

#Close Connection
#connection.close

########Windows#######

root = Tk()
root.title("Softmaking")
root.resizable(False,False)
#root.iconbitmap("logo-sm.ico")

#Center windows
app_width = 600
app_height = 500

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x = (screen_width/2) - (app_width/2)
y = (screen_height/2) - (app_height/2)

root.geometry(f'{app_width}x{app_height}+{int(x)}+{int(y)}')



#Def
def printLabel():
    label = Label(root, text="Yeeey " + fakeInput.get())
    label.pack()

def popup():
    messagebox.showinfo(message="yeeeeeey messageboxx!!!")

def folderSelect():
    folder_selected = filedialog.askdirectory()
    print(folder_selected)

#Def Windows 2
def openConfigurationTk():
    global imagen
    #Config tk
    configurationTk = Toplevel()
    configurationTk.resizable(False,False)
    configurationTk.protocol("WM_DELETE_WINDOW", exit)
    configurationTk.title("Configuraciones")
    configurationTk.overrideredirect(True)

    imagen = ImageTk.PhotoImage(Image.open("valorant.jpg"))

    width = imagen.width()
    height = (imagen.height() + 100)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width/2) - (app_width/2)
    y = (screen_height/2) - (app_height/2)

    configurationTk.geometry(f'{width}x{height}+{int(x)}+{int(y)}')
    labelimagen = Label(configurationTk, image=imagen)
    labelimagen.pack()

    buton = Button(configurationTk, text="Cambiar directorio", command=folderSelect)
    buton.pack()

    closeWindow = Button(configurationTk, text="Cerrar Ventana", command=configurationTk.destroy)
    closeWindow.pack()

#Buttons
buttonFlase = Button(root, text="yeeey", command=printLabel).pack()
configButton = Button(root, text="Configuraciones", command=openConfigurationTk, fg="blue")
configButton.pack()

messagebuton = Button(root, text="Popup", command=popup).pack()

exitButton = Button(root, text="Salir", command=root.quit)
exitButton.pack()
#Inputs
fakeInput = Entry(root)
fakeInput.pack()
fakeInput.insert(0, "Escribe algoo1!!")

#Images
#imagen = ImageTk.PhotoImage(Image.open("valorant.jpg"))
#labelimagen = Label(image=imagen)
#labelimagen.pack()

#Frames
frame = LabelFrame(root, text="yeeey frame!!")
frame.pack()
b = Button(frame, text="Este es un frame :o")
b.pack()

root.mainloop()