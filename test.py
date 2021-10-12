# import builtins
# import cv2
# from PIL import Image, ImageTk
# from tkinter import *
# import imutils

# def visualizar():
#     global cap
#     print('visualizar')
#     if cap is not None:
#         print('cap?')
#         ret, frame = cap.read()
#         if ret == True:
#             print('ret?')
#             frame = imutils.resize(frame, width=640)
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             im = Image.fromarray(frame)
#             img = ImageTk.PhotoImage(image=im)

#             labelVideo.configure(image=img)
#             # labelVideo.image = img
#             labelVideo.after(10, visualizar)
#         else:
#             # labelVideo.image = ''
#             cap.release()

# def iniciar():
#     print('llega?')
#     global cap
#     cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
#     visualizar()

# cap = None
# root = Tk()

# btnIniciar = Button(root, text='Iniciar', width=45, command=iniciar)
# btnIniciar.grid(column=0, row=0, padx=5, pady=5)

# btnFinalizar = Button(root, text='Finalizar', width=45)
# btnFinalizar.grid(column=1, row=0, padx=5, pady=5)

# labelVideo = Label(root)
# labelVideo.grid(row=1, column=0, columnspan=2)

# root.mainloop()


###########################################################################################################################

# import tkinter as tk
# import time
# import threading as th
# root = tk.Tk()
# value = 200
# ph = tk.DoubleVar()
# f = open('datos_mv.txt', 'a')
# f.write(str(value) + "\n")
# f.close()

# def funcion_ph():
#     while -400 <= value <= 400:
#         time.sleep(2)
#         x = 7 - (value / 57.14)
#         y = 5.8
#         print(x)
# def click_button(h):
#     if h == 1:
#         boton1.config(state='disabled')
#         boton2.config(state='normal')
#         t1 = th.Thread(target=funcion_ph)
#         t1.start()
#     if h == 2:
#         boton2.config(state='disabled')  # este boton es solo para probar si el primer boton se habilita
#         boton1.config(state='normal')
#         print(2)
# boton1 = tk.Button(root, text='Acidez',
#                    fg='white', bg='dodger blue', activebackground='deep sky blue3',
#                    activeforeground='white', width=10, height=1, font='Calibri, 13',
#                    command=lambda: click_button(1))
# boton1.place(x=20, y=60)
# boton2 = tk.Button(root, text='Conductividad',
#                    fg='white', bg='dodger blue', activebackground='deep sky blue3',
#                    activeforeground='white', width=11, height=1, font='Calibri, 13',
#                    command=lambda: click_button(2))
# boton2.place(x=160, y=60)
# root.mainloop()

M1 = [[8, 14, -6], [12,7,4], [-11,3,21]]

print(M1)
print(M1[0][0])

for i in range(len(M1)):
    for j in range(len(M1[i])):
        print(M1[i][j])