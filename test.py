from tkinter import Entry, Tk

root = Tk()
root.title('test')
root.geometry("800x500")

entry = Entry()
# Posicionarla en la ventana.
entry.place(x=50, y=50)


root.mainloop()