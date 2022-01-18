import tkinter as tk
from tkinter import CENTER, END, Listbox, Tk, ttk
import requests

url = 'http://127.0.0.1:8000/'

def userList(token, refresh=None):
    credentials = {'refresh': refresh}
    request = requests.get(url + 'users/', data = credentials, headers={'Authorization': 'Bearer ' + token})
    request_dictionary = request.json()
    return request_dictionary


root = Tk()
root.title('TreeView')
root.geometry("800x500")

treeViewTest = ttk.Treeview(root)

#Columnas
treeViewTest['columns'] = ("Nombre de usuario", "Nombre", "Apellido", "E-mail", "Ultima conexión")

#Configurar columnas
treeViewTest.column("#0", width=120, minwidth=25)
treeViewTest.column("Nombre de usuario", anchor=tk.W, width=120)
treeViewTest.column("Nombre", anchor=tk.W, width=120)
treeViewTest.column("Apellido", anchor=tk.W, width=120)
treeViewTest.column("E-mail", anchor=tk.W, width=160)
treeViewTest.column("Ultima conexión", anchor=tk.W, width=120)

#Crear Encabezados
treeViewTest.heading("#0", text="Label", anchor=tk.W)
treeViewTest.heading("Nombre de usuario",text="Nombre de usuario", anchor=tk.W)
treeViewTest.heading("Nombre",text="Nombre", anchor=tk.W)
treeViewTest.heading("Apellido",text="Apellido", anchor=tk.W)
treeViewTest.heading("E-mail",text="E-mail", anchor=tk.W)
treeViewTest.heading("Ultima conexión",text="Ultima conexión", anchor=tk.W)

# Agregar datos
userListTest = userList('eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjQyMjEwOTU0LCJpYXQiOjE2NDIxNzg1NTQsImp0aSI6ImVhMWFhNTU4NTY5MDQ5ZGM5YTUxYWUwNzBkNzQyODczIiwidXNlcl9pZCI6MX0._6_RqXlXHGtyZtFvLzvBLTrOiAqnrNGVh9tZxZ09aLk')
print(userListTest)
count = 0
for record in userListTest:
    treeViewTest.insert(parent='', index='end', iid=count, text="Parent", values=(record['username'], record['name'], record['last_name'], record['email'], record['last_login']))
    count += 1

# treeViewTest.insert(parent='', index='end', iid=0, text="Parent", values=("username", "jhon", "malo", "emailmlao", "ultima conexcion"))
# treeViewTest.insert(parent='', index='end', iid=1, text="Parent", values=("username", "test", "fdsf", "examaple@xpmna'lew", "ultima conexcion"))

treeViewTest.pack(pady=20)

# # Lisbox
# testListbox = Listbox(root)
# testListbox.pack(pady=15)

# testListbox.insert(END, "user")
# testListbox.insert(END, "user2")
# testListbox.insert(END, "user3")
# testList = ["user4", "user5", "user6"]

# userListTest = userList('eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjQyMjEwOTU0LCJpYXQiOjE2NDIxNzg1NTQsImp0aSI6ImVhMWFhNTU4NTY5MDQ5ZGM5YTUxYWUwNzBkNzQyODczIiwidXNlcl9pZCI6MX0._6_RqXlXHGtyZtFvLzvBLTrOiAqnrNGVh9tZxZ09aLk')
# print(userListTest)

# for item in testList:
#     testListbox.insert(END, item)

# for item in userListTest:
#     testListbox.insert(END, item['username'])
#     print(item['id'])

root.mainloop()

# import tkinter as tk
# from tkinter import ttk
# class Application(ttk.Frame):
    
#     def __init__(self, main_window):
#         super().__init__(main_window)
#         main_window.title("Lista en Tcl/Tk")
        
#         self.listbox = tk.Listbox(self)
#         self.listbox.pack()
#         self.pack()
# main_window = tk.Tk()
# app = Application(main_window)
# app.mainloop()