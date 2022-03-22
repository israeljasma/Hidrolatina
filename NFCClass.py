from smartcard.CardRequest import CardRequest
from smartcard.Exceptions import CardRequestTimeoutException
from smartcard.CardType import AnyCardType
from smartcard import util
from threading import Thread
import concurrent.futures
from tkinter import messagebox
from Services import API_Services
from UserClass import Person
import time

class NFC():
    def __init__(self, ventana, ventanaNext):
        self.tktest = ventana()
        self.thread = Thread(target=self.identify, args=(ventanaNext,), daemon=True)
        self.thread.start()

    
    def identify(self, ventanaNext):
        uid = None
        WAIT_FOR_SECONDS = 60
        # respond to the insertion of any type of smart card
        card_type = AnyCardType()

        # create the request. Wait for up to x seconds for a card to be attached
        request = CardRequest(timeout=WAIT_FOR_SECONDS, cardType=card_type)

        while True:
            # listen for the card
            service = None
            try:
                service = request.waitforcard()
            except CardRequestTimeoutException:
                print("Tarjeta no detectada")
                # could add "exit(-1)" to make code terminate

            # when a card is attached, open a connection
            try:
                conn = service.connection
                conn.connect()

                # get the ATR and UID of the card
                get_uid = util.toBytes("FF CA 00 00 00")
                data, sw1, sw2 = conn.transmit(get_uid)
                uid = util.toHexString(data)
                # print(uid)
                try:
                    loginNFC = API_Services.loginNFC(uid)
                except:
                    print('no hay servidor')
                    self.stop()
                    messagebox.showerror(title='Error de conexión', message='No se ha podido establecer una conexión con el servidor. Comuníquese con su encargado de TI.')
                    break
                # status = util.toHexString([sw1, sw2])
                # if uid == "44 CE 4A 0B":
                if 'token' in loginNFC:
                    user = Person(loginNFC['user']['username'], loginNFC['user']['name'], loginNFC['user']['last_name'], loginNFC['user']['email'], loginNFC['token'], loginNFC['refresh-token'])
                    print(user)
                    break
                    #Hide Root Window
                    # root.withdraw()
                else:
                    messagebox.showinfo(message=loginNFC['error'], title="Login")
            except:
                pass

        # time.sleep(2)
        # self.tktest.after(4000, self.stop)
        
        ventanaNext(user)
        self.stop()
        # self.tktest.after(2000,self.stop)
        
        
        
        
        
        # ventanaNext(user)


    def stop(self):
        self.tktest.destroy()
        try:
            self.thread.join()
        except:
            pass

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.thread.join()
        except:
            pass

class adminNFC():
    def __init__(self):

    #     self.thread = Thread(target=self.readNFC, args=(), daemon=True)
    #     self.thread.start()


        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.readNFC)
            return_value = future.result()
            print(return_value)

    def readNFC(self):
        uid = None
        WAIT_FOR_SECONDS = 60
        # respond to the insertion of any type of smart card
        card_type = AnyCardType()

        # create the request. Wait for up to x seconds for a card to be attached
        request = CardRequest(timeout=WAIT_FOR_SECONDS, cardType=card_type)

        while True:
            # listen for the card
            service = None
            try:
                service = request.waitforcard()
            except CardRequestTimeoutException:
                print("Tarjeta no detectada")
                # could add "exit(-1)" to make code terminate

            # when a card is attached, open a connection
            try:
                conn = service.connection
                conn.connect()

                # get the ATR and UID of the card
                get_uid = util.toBytes("FF CA 00 00 00")
                data, sw1, sw2 = conn.transmit(get_uid)
                uid = util.toHexString(data)
                break
            except:

                pass

        return uid

    # def getuid():
    #     WAIT_FOR_SECONDS = 60
    #     # respond to the insertion of any type of smart card
    #     card_type = AnyCardType()

    #     # create the request. Wait for up to x seconds for a card to be attached
    #     request = CardRequest(timeout=WAIT_FOR_SECONDS, cardType=card_type)

    #     while True:
    #         # listen for the card
    #         service = None
    #         try:
    #             service = request.waitforcard()
    #         except CardRequestTimeoutException:
    #             print("Tarjeta no detectada")
    #             # could add "exit(-1)" to make code terminate

    #         # when a card is attached, open a connection
    #         try:
    #             conn = service.connection
    #             conn.connect()

    #             # get the ATR and UID of the card
    #             get_uid = util.toBytes("FF CA 00 00 00")
    #             data, sw1, sw2 = conn.transmit(get_uid)
    #             uid = util.toHexString(data)
    #             status = util.toHexString([sw1, sw2])
    #             print(uid)
    #             break
    #         except:
    #             pass
    #     time.sleep(2)
    #     return


# NFC.identify()
# from multiprocessing import Queue
# from threading import Thread

# def foo(bar):
#     print('hello {0}'.format(bar))
#     return 'hola'

# que = Queue()

# t = Thread(target=lambda q, arg1: q.put(foo(arg1)), args=(que, 'world!'))
# t.start()
# t.join()
# result = que.get()
# print(result)
# import time
# from threading import Thread
# from tkinter import *


# root = Tk()
# root.title("Softmaking")
# root.resizable(False,False)

# #Center windows
# app_width = 300
# app_height = 300

# screen_width = root.winfo_screenwidth()
# screen_height = root.winfo_screenheight()

# x = (screen_width/2) - (app_width/2)
# y = (screen_height/2) - (app_height/2)

# root.geometry(f'{app_width}x{app_height}+{int(x)}+{int(y)}')

# def ReadNFC(res):
#     from smartcard.CardRequest import CardRequest
#     from smartcard.Exceptions import CardRequestTimeoutException
#     from smartcard.CardType import AnyCardType
#     from smartcard import util

#     WAIT_FOR_SECONDS = 60
#     # respond to the insertion of any type of smart card
#     card_type = AnyCardType()

#     # create the request. Wait for up to x seconds for a card to be attached
#     request = CardRequest(timeout=WAIT_FOR_SECONDS, cardType=card_type)

#     while True:
#         # listen for the card
#         service = None
#         try:
#             service = request.waitforcard()
#         except CardRequestTimeoutException:
#             print("Tarjeta no detectada")
#             # could add "exit(-1)" to make code terminate

#         # when a card is attached, open a connection
#         try:
#             conn = service.connection
#             conn.connect()

#             # get the ATR and UID of the card
#             get_uid = util.toBytes("FF CA 00 00 00")
#             data, sw1, sw2 = conn.transmit(get_uid)
#             uid = util.toHexString(data)
#             status = util.toHexString([sw1, sw2])
#             if uid == "44 CE 4A 0B":

#             # print the ATR and UID of the card
#             # print("ATR = {}".format(util.toHexString(conn.getATR())))
#                 print("Operador Reconocido")
#                 break
#         except:
#             pass
#     time.sleep(2)
#     res.set("Abriendo")

# def nfc_identify():
#     # import time
#     # from threading import Thread


#     #NFC_TEST
#     global NFCTestTk
#     NFCTestTk = Toplevel()
#     # NFCTestTk.title('Imagenes')
#     # NFCTestTk.resizable(False,False)
#     NFCTestTk.protocol("WM_DELETE_WINDOW", exit)
#     # NFCTestTk.overrideredirect(True)
#     # NFCTestTk.geometry('300x300')

#     width = 300
#     height = 300
#     screen_width = root.winfo_screenwidth()
#     screen_height = root.winfo_screenheight()

#     x = (screen_width/2) - (app_width/2)
#     y = (screen_height/2) - (app_height/2)

#     NFCTestTk.geometry(f'{width}x{height}+{int(x)}+{int(y)}')
#     v = StringVar()
#     v.set("Esperando Identificación ....")
#     Label(NFCTestTk,textvariable=v).pack(pady = 100)
#     # Label(NFCTestTk,text ="nowwwwwwwww").pack(pady = 100)

#     # root.withdraw()
#     Button(NFCTestTk, text="Cerrar Ventana", command=lambda:closeTk()).pack(pady=10)


#     thread= Thread(target=ReadNFC, args=(v,))
#     thread.start()
#     # thread.join()
    

#     def closeTk():
#         NFCTestTk.destroy()
#         root.deiconify()


# testButton = Button(root, text='Test download',command=nfc_identify, fg='red').pack()

# exitButton = Button(root, text="Salir", command=root.quit)
# exitButton.pack()

# root.mainloop()