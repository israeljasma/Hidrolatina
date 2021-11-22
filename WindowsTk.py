import time
from datetime import datetime, timedelta
from tkinter import *
from tkinter import messagebox, filedialog, simpledialog, Listbox
from PIL import ImageTk, Image

from Services import API_Services
from UserClass import Person
from imagenClipClass import imageClip
from FileManagementClass import FileManagement
from CameraStream import CameraStream
from PpeDetector import PpeDetector
import cv2

class WindowsTk:

    def __init__(self):
        self.ppedet = PpeDetector()

    def loadALL(self):

        # self.ppedet = PpeDetector()
        # from threading import Thread
        # libThread= Thread(target=librerias, args=(),daemon=True)
        # libThread.start()
        self.model_mdetr, self.transform = self.ppedet.importMDETR().init()
        self.ppedet.loadClip()
        self.ppedet.loadEfficientDet()
        messagebox.showinfo(message="Dependencias cargadas")


    ########Windows#######

    #Def
    def popup(self, message):
        messagebox.showinfo(message=message)

    def folderSelect(self):
        folder_selected = filedialog.askdirectory()
        print(folder_selected)

    def folderframeSelect(self):
        # global frame_selected
        self.frame_selected = filedialog.askopenfilename()
        print(self.frame_selected)

    ###################Def Windows's###################

    def showPytorchCameraTk(self, user):
        # import datetime
        import numpy as np

        #Var/Global
        # global det
        # global image
        # global original_image
        self.det=0

        #Tkinter config
        pytorchCameraTk = Toplevel()
        pytorchCameraTk.title('Camara')
        # pytorchCameraTk.resizable(False,False)
        pytorchCameraTk.config(background="#cceeff")
        pytorchCameraTk.overrideredirect(True)
        pytorchCameraTk.geometry(f'{pytorchCameraTk.winfo_screenwidth()}x{pytorchCameraTk.winfo_screenheight()}')
        # pytorchCameraTk.geometry(f'{1280}x{720}')
    
        # pytorchCameraTk.geometry("1280x720")

        self.image = PhotoImage(file="white-image.png")
        self.original_image = self.image.subsample(1,1)

        #Frame Camera
        cameraFrame = Frame(pytorchCameraTk, width=pytorchCameraTk.winfo_screenwidth()*0.7, height=pytorchCameraTk.winfo_screenheight(), bg='#cceeff')
        cameraFrame.grid(row=0, column=0)

        # #Frame detections
        detectionFrame = Frame(pytorchCameraTk, bg="#cceeff", width=pytorchCameraTk.winfo_screenwidth()*0.3, height=pytorchCameraTk.winfo_screenheight())
        detectionFrame.grid(row=0, column=1)

        # ##Subs frames lvl 1 detections
        headFrame = Frame(detectionFrame, bg="#b5e6ff", width=detectionFrame.winfo_reqwidth(), height=detectionFrame.winfo_reqheight()*0.334)
        headFrame.grid(row=0, column=0)

        handFrame = Frame(detectionFrame, bg="#b5e6ff", width=detectionFrame.winfo_reqwidth(), height=detectionFrame.winfo_reqheight()*0.334)
        handFrame.grid(row=1, column=0)

        bootFrame = Frame(detectionFrame, bg="#b5e6ff", width=detectionFrame.winfo_reqwidth(), height=detectionFrame.winfo_reqheight()*0.334)
        bootFrame.grid(row=2, column=0)

        # ###Subs frames lvl 2 headFrame
        imageHeadFrame = Frame(headFrame, bg="#b5e6ff", width=headFrame.winfo_reqwidth()*0.5, height=headFrame.winfo_reqheight())
        imageHeadFrame.grid(row=0, column=0)

        dataHeadFrame = Frame(headFrame, bg="#b5e6ff", width=headFrame.winfo_reqwidth()*0.5, height=headFrame.winfo_reqheight())
        dataHeadFrame.grid(row=0, column=1)

        # ###Subs frames lvl 2 handFrame
        imageHandFrame = Frame(handFrame, bg="#b5e6ff", width=handFrame.winfo_reqwidth()*0.5, height=handFrame.winfo_reqheight())
        imageHandFrame.grid(row=0, column=0)

        dataHandFrame = Frame(handFrame, bg="#b5e6ff", width=handFrame.winfo_reqwidth()*0.5, height=handFrame.winfo_reqheight())
        dataHandFrame.grid(row=0, column=1)

        # ###Subs frames lvl 2 bootFrame
        imageBootFrame = Frame(bootFrame, bg="#b5e6ff", width=bootFrame.winfo_reqwidth()*0.5, height=bootFrame.winfo_reqheight())
        imageBootFrame.grid(row=0, column=0)

        dataBootFrame = Frame(bootFrame, bg="#b5e6ff", width=bootFrame.winfo_reqwidth()*0.5, height=bootFrame.winfo_reqheight())
        dataBootFrame.grid(row=0, column=1)

        # ###Label imageHeadFrame Sub frame lvl 2 headFrame
        Label(imageHeadFrame, image=self.original_image).grid(row=1, column=0, padx=5, pady=5)

        # ###Label imageHandFrame Sub frame lvl 2 handFrame
        Label(imageHandFrame, image=self.original_image).grid(row=1, column=0, padx=5, pady=5)

        # ###Label imagebootFrame Sub frame lvl 2 bootFrame
        Label(imageBootFrame, image=self.original_image).grid(row=1, column=0, padx=5, pady=5)

        # ####Label dataHeadFrame Sub frame lvl 2 headFrame
        Label(dataHeadFrame, text="Casco", width=8).grid(row=0, column=0, padx=5, pady=5)
        Label(dataHeadFrame, width=10).grid(row=0, column=1, padx=5, pady=5)

        Label(dataHeadFrame, text="Audífonos", width=8).grid(row=1, column=0, padx=5, pady=5)
        Label(dataHeadFrame, width=10).grid(row=1, column=1, padx=5, pady=5)

        Label(dataHeadFrame, text="Antiparras", width=8).grid(row=2, column=0, padx=5, pady=5)
        Label(dataHeadFrame, width=10).grid(row=2, column=1, padx=5, pady=5)

        Label(dataHeadFrame, text="Mascarilla", width=8).grid(row=3, column=0, padx=5, pady=5)
        Label(dataHeadFrame, width=10).grid(row=3, column=1, padx=5, pady=5)

        # ####Label dataHandFrame Sub frame lvl 2 handFrame
        Label(dataHandFrame, text="Guantes", width=8).grid(row=0, column=0, padx=5, pady=5)
        Label(dataHandFrame, width=10).grid(row=0, column=1, padx=5, pady=5)

        # ####Label dataBootFrame Sub frame lvl 2 bootFrame
        Label(dataBootFrame, text="Botas", width=8).grid(row=0, column=0, padx=5, pady=5)
        Label(dataBootFrame, width=10).grid(row=0, column=1, padx=5, pady=5)

        #Capture video frames
        labelVideo = Label(cameraFrame)
        labelVideo.grid(row=0, column=0)
        cap = CameraStream().start()
        # cap = cv2.VideoCapture(0)

        camWidth = round(cameraFrame.winfo_reqwidth())
        camHeight = round(cameraFrame.winfo_reqheight()*0.85)

        #Def into tk
        def closeTk(self):
            #Destroy window
            cap.stop()
            pytorchCameraTk.destroy()
            # root.deiconify()
        
        def showFrame(self):
            # _, frame = cap.read()
            # frame = cv2.flip(frame, 1)
            try:
                frame=cv2.imread(self.frame_selected)
            except:
                frame = cap.read()

            out = self.ppedet.efficientDet(frame)
            ori_img = frame.copy()

            for j in range(len(out['bbox'])):
                (x1, y1, x2, y2) = out['bbox'][j].astype(np.int)
                cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                # obj = obj_list[out['class_ids'][j]]
                score = float(out['scores'][j])

                cv2.putText(ori_img, '{:.3f}'.format(score),
                            (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, .5,
                            (255, 255, 0), 2)

            cv2image = cv2.cvtColor(cv2.resize(ori_img, (600, 500)), cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            labelVideo.imgtk = imgtk
            labelVideo.configure(image=imgtk)

            print(self.det)
            if len(out['class_ids']) == 0:
                self.det = 0
            if len(out['class_ids']) > 0:
                self.det += 1
                if self.det==20:

                    print("Reset")
                    for i in range((out['scores']).size):
                        detected_boxes= out['bbox'][i]

                    # Crop and save detedtec bounding box image

                    xmin = int((detected_boxes[0]))
                    ymin = int((detected_boxes[1]))
                    xmax = int((detected_boxes[2]))
                    ymax = int((detected_boxes[3]))
                    cropped_img =frame[ymin:ymax,xmin:xmax]

                    im = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                    cap.stop()
                    copy_imgtk = imgtk
                    labelVideo.imgtk = copy_imgtk
                    mdetr_list=self.ppedet.MDETR(self.model_mdetr, self.transform, im)
                    print(mdetr_list)
                    self.listImagenClip = []
                    for bodypart in mdetr_list.keys(): 
                        self.listImagenClip.append(imageClip(self.ppedet.names_ppe[bodypart], ImageTk.PhotoImage(mdetr_list[bodypart].resize((150,150))), self.ppedet.clip(bodypart, mdetr_list)))
                    
                    updateLabel(self)
                    return self
        ################################################ CORRERGIR ###############################################
        ################################################ CORRERGIR ###############################################
        ################################################ CORRERGIR ###############################################
                    
            if self.det<20:
                labelVideo.after(10, showFrame, self)
        
        def counterPopUp(self, endTime, booleanAnswer):
            if datetime.now() > endTime:
                print('si')
                print(datetime.now().strftime('%H:%M:%S'), endTime.strftime('%H:%M:%S'))
                print('funciona')
                self.popupIdentificationTk(booleanAnswer)
            else:
                print('no')
                print(datetime.now().strftime('%H:%M:%S'), endTime.strftime('%H:%M:%S'))
                pytorchCameraTk.after(5000, counterPopUp, self, endTime, booleanAnswer)

        def updateLabel(self):
            #Head Frame
            Label(imageHeadFrame, image=(self.listImagenClip[0].getImage())).grid(row=1, column=0, padx=5, pady=5)
            Label(dataHeadFrame, text=(self.listImagenClip[0].getAnswer()[0]), width=15).grid(row=0, column=1, padx=5, pady=5)
            Label(dataHeadFrame, text=(self.listImagenClip[0].getAnswer()[1]), width=15).grid(row=1, column=1, padx=5, pady=5)
            Label(dataHeadFrame, text=(self.listImagenClip[0].getAnswer()[2]), width=15).grid(row=2, column=1, padx=5, pady=5)
            Label(dataHeadFrame, text=(self.listImagenClip[0].getAnswer()[3]), width=15).grid(row=3, column=1, padx=5, pady=5)

            #Hand Frame
            Label(imageHandFrame, image=(self.listImagenClip[1].getImage())).grid(row=1, column=0, padx=5, pady=5)
            Label(dataHandFrame,  text=(self.listImagenClip[1].getAnswer()[0]), width=15).grid(row=0, column=1, padx=5, pady=5)

            #Boot Frame
            Label(imageBootFrame, image=(self.listImagenClip[2].getImage())).grid(row=1, column=0, padx=5, pady=5)
            Label(dataBootFrame, text=(self.listImagenClip[2].getAnswer()[0]), width=15).grid(row=0, column=1, padx=5, pady=5)

            booleanAnswer = None
            for list in self.listImagenClip:
                for j in range(len(list.getAnswer())):
                    if list.getAnswer()[j] == 'OK':
                        print(list.getName()[j])
                        print(list.getAnswer()[j])
                        booleanAnswer = True
                    else:
                        booleanAnswer = False
                    print(booleanAnswer)

            endTime = datetime.now() + timedelta(seconds=10)
            if len(self.listImagenClip) > 0:
                counterPopUp(self, endTime, booleanAnswer)
            # pytorchCameraTk.after(1, counterPopUp, endTime, booleanAnswer)

        exitButton = Button(pytorchCameraTk, text='Cerrar ventana', command=lambda:closeTk(self))
        exitButton.grid(row=1, column=0)

        testButtonUpdate = Button(pytorchCameraTk, text='Test Update', command=lambda:updateLabel(self))
        testButtonUpdate.grid(row=1, column=1)
        showFrame(self)


    def configCameraTk(self, configurationTk):
        # Config tk
        configCameraTk = Toplevel()
        configCameraTk.resizable(False,False)
        configCameraTk.protocol("WM_DELETE_WINDOW", exit)
        configCameraTk.title("Configuracion camaras")
        # configCameraTk.overrideredirect(True)

        width = 200
        height = 300
        screen_width = 200
        screen_height = 300
        # screen_width = root.winfo_screenwidth()
        # screen_height = root.winfo_screenheight()

        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)

        configCameraTk.geometry(f'{width}x{height}+{int(x)}+{int(y)}')

        #Def
        def closeTk(configurationTk):
            configCameraTk.destroy()
            configurationTk.deiconify()

        def setCamera():
            global varCamera
            varCamera = simpledialog.askstring(title="Camara", prompt="Ingrese Camara:")
            if varCamera == '':
                varCamera = 0
                print(varCamera)
            else:
                print(varCamera)

        #Hide configurationTk Window
        configurationTk.withdraw()

        #Labels Tk
        buttonClass = Button(configCameraTk, text="Configurar Camara(Principal)", command=lambda:setCamera())
        buttonClass.pack()

        #Buttons Tk
        closeWindow = Button(configCameraTk, text="Cerrar Ventana", command=lambda:closeTk(configurationTk))
        closeWindow.pack()

    def nfc_identifyTk(self):
        # import concurrent.futures
        # Config tk
        NFC_Tk = Toplevel()
        # NFC_Tk.resizable(False,False)
        NFC_Tk.title("Identificación")
        # NFC_Tk.overrideredirect(True)
        NFC_Tk.geometry(f'{NFC_Tk.winfo_screenwidth()}x{NFC_Tk.winfo_screenheight()}')

        #Def
        def time_string():
            return time.strftime('%H:%M:%S')

        def update():
            timeLabel.configure(text=time_string())
            # Recursive
            timeLabel.after(1000, update)

        # def thread_identify():
        #     test = NFC.identify()
        #     print(test)
        #     time.sleep(1)
        #     thread_identify()

            # while True:
            # NFC.identify()
            # thread = Thread(target=NFC.identify, args=())
            # thread.start()
            # var = thread.join()
            # print(var)
            # thread.start()
            # return True

        def closeTk():
            NFC_Tk.destroy()
            # root.deiconify()
        
        #Hide Root Window
        # root.withdraw()

        # Frame
        NFCFrame = Frame(NFC_Tk, width=NFC_Tk.winfo_screenwidth(), height=NFC_Tk.winfo_screenheight(), bg='#CCEEFF')
        NFCFrame.grid()

        # Create left and right frames
        left_frame = Frame(NFCFrame, width=round(NFCFrame.winfo_reqwidth()*0.5), height=round(NFCFrame.winfo_reqheight()), bg='#CCEEFF')
        left_frame.grid(row=0, column=0)

        right_frame = Frame(NFCFrame, width=round(NFCFrame.winfo_reqwidth()*0.5), height=round(NFCFrame.winfo_reqheight()), bg='#CCEEFF')
        right_frame.grid(row=0, column=1)

        # Divide right frame
        up_frame_right_frame = Frame(right_frame, width=right_frame.winfo_reqwidth(), height=right_frame.winfo_reqheight()*0.15, bg='#CCEEFF')
        up_frame_right_frame.grid(row=0, column=0)

        down_frame_right_frame = Frame(right_frame, width=right_frame.winfo_reqwidth(), height=right_frame.winfo_reqheight()*0.85, bg='#CCEEFF')
        down_frame_right_frame.grid(row=1, column=0)

        # Labels left_frame
        global imageWaitDetectionLeft
        imageWaitDetectionLeft = Image.open("images/waiting_identification_left.png")
        imageWaitDetectionLeft = imageWaitDetectionLeft.resize((round(NFCFrame.winfo_reqwidth()*0.5), round(NFCFrame.winfo_reqheight())), Image.ANTIALIAS)
        imageWaitDetectionLeft = ImageTk.PhotoImage(imageWaitDetectionLeft)
        imageLabelLeft_Frame = Label(left_frame, image=imageWaitDetectionLeft, borderwidth=0)
        imageLabelLeft_Frame.grid(row=0, column=0)

        global imageWaitDetectionRight
        imageWaitDetectionRight = Image.open('images/waiting_identification_right.png')
        imageWaitDetectionRight = imageWaitDetectionRight.resize((round(down_frame_right_frame.winfo_reqwidth()), round(down_frame_right_frame.winfo_reqheight())), Image.ANTIALIAS)
        imageWaitDetectionRight = ImageTk.PhotoImage(imageWaitDetectionRight)
        imageLabelRight_frameDown = Label(right_frame, image=imageWaitDetectionRight, borderwidth=0)
        imageLabelRight_frameDown.grid(row=1, column=0)

        timeLabel = Label(up_frame_right_frame, text=time_string(), bg='#CCEEFF', font=('Digital-7', up_frame_right_frame.winfo_reqheight()))
        timeLabel.grid()
        
        timeLabel.after(1000, update)

        # thread= Thread(target=identify, args=())
        # thread.start()
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #         future = executor.submit(NFC.identify)
                # return_value = future.result()
                # print(return_value)

        # NFC_Tk.after(3000, thread_identify)

        #Buttons Tk
        # Button(NFC_Tk, text="Cerrar Ventana", command=lambda:closeTk()).pack(pady=10)

        # thread= Thread(target=NFC.identify, args=())
        # thread.start()

        # thread.join()

        return NFC_Tk

    def popupIdentificationTk(self, booleanAnswer):
        # Config tk
        popupIdentificationTk = Toplevel()
        popupIdentificationTk.resizable(False,False)
        popupIdentificationTk.after(10000, popupIdentificationTk.destroy)
        popupIdentificationTk.overrideredirect(True)
        popupIdentificationTk.geometry(f'{popupIdentificationTk.winfo_screenwidth()}x{popupIdentificationTk.winfo_screenheight()}')
        # popupIdentificationTk.geometry("1280x720")

        #Code

        PopUpIdentificationFrame = Frame(popupIdentificationTk, width=popupIdentificationTk.winfo_screenwidth(), height=popupIdentificationTk.winfo_screenheight(), bg='#CCEEFF')
        PopUpIdentificationFrame.grid()

        if booleanAnswer:
            global detections
            detections = Image.open('images/approved_detections.png')
            detections = detections.resize((PopUpIdentificationFrame.winfo_reqwidth(), PopUpIdentificationFrame.winfo_reqheight()), Image.ANTIALIAS)
            detections = ImageTk.PhotoImage(detections)

            imageFrame = Frame(PopUpIdentificationFrame, width=PopUpIdentificationFrame.winfo_reqwidth(), height=PopUpIdentificationFrame.winfo_reqheight())
            imageFrame.grid(row=0, column=0)
            imageLabel = Label(imageFrame, image=detections)
            imageLabel.pack()
        
        else:
            global imageTop, imageMiddleLeft, imageMiddleRight, imageBottom
            # Create top, middle and bottom frames
            top_frame = Frame(PopUpIdentificationFrame, width=round(PopUpIdentificationFrame.winfo_reqwidth()), height=round(PopUpIdentificationFrame.winfo_reqheight()*0.19), bg='#CCEEFF')
            top_frame.grid(row=0, column=0)
        
            middle_frame = Frame(PopUpIdentificationFrame, width=round(PopUpIdentificationFrame.winfo_reqwidth()), height=round(PopUpIdentificationFrame.winfo_reqheight()*0.51), bg='#CCEEFF')
            middle_frame.grid(row=1, column=0)

            bottom_frame = Frame(PopUpIdentificationFrame, width=round(PopUpIdentificationFrame.winfo_reqwidth()), height=round(PopUpIdentificationFrame.winfo_reqheight()*0.3), bg='#CCEEFF')
            bottom_frame.grid(row=2, column=0)

            # # Divide middle frame
            left_middle_frame = Frame(middle_frame, width=middle_frame.winfo_reqwidth()*0.4, height=middle_frame.winfo_reqheight(), bg='#CCEEFF')
            left_middle_frame.grid(row=0, column=0)

            right_middle_frame = Frame(middle_frame, width=middle_frame.winfo_reqwidth()*0.6, height=middle_frame.winfo_reqheight(), bg='#CCEEFF')
            right_middle_frame.grid(row=0, column=1)

            # Labels Top Frame
            imageTop = Image.open('images/unapproved_detections_top.png')
            imageTop = imageTop.resize((top_frame.winfo_reqwidth(), top_frame.winfo_reqheight()), Image.ANTIALIAS)
            imageTop = ImageTk.PhotoImage(imageTop)

            imageMiddleLeft = Image.open('images/unapproved_detections_middle_left.png')
            imageMiddleLeft = imageMiddleLeft.resize((left_middle_frame.winfo_reqwidth(), left_middle_frame.winfo_reqheight()), Image.ANTIALIAS)
            imageMiddleLeft = ImageTk.PhotoImage(imageMiddleLeft)

            imageMiddleRight = Image.open('images/unapproved_detections_middle_right.png')
            imageMiddleRight = imageMiddleRight.resize((right_middle_frame.winfo_reqwidth(), right_middle_frame.winfo_reqheight()), Image.ANTIALIAS)
            imageMiddleRight = ImageTk.PhotoImage(imageMiddleRight)

            imageBottom = Image.open('images/unapproved_detections_bottom.png')
            imageBottom = imageBottom.resize((bottom_frame.winfo_reqwidth(), bottom_frame.winfo_reqheight()), Image.ANTIALIAS)
            imageBottom = ImageTk.PhotoImage(imageBottom)

            imageTopLabel = Label(top_frame, image=imageTop, width=top_frame.winfo_reqwidth(), height=top_frame.winfo_reqheight(), borderwidth=0)
            imageTopLabel.grid(row=0, column=0)

            imageMiddleLeftLabel = Label(left_middle_frame, image=imageMiddleLeft, width=left_middle_frame.winfo_reqwidth(), height=left_middle_frame.winfo_reqheight(), borderwidth=0)
            imageMiddleLeftLabel.grid(row=0, column=0)

            imageMiddleRightLabel = Label(right_middle_frame, image=imageMiddleRight, width=right_middle_frame.winfo_reqwidth(), height=right_middle_frame.winfo_reqheight(), borderwidth=0)
            imageMiddleRightLabel.grid(row=0, column=0)

            imageBottomLabel = Label(bottom_frame, image=imageBottom, width=bottom_frame.winfo_reqwidth(), height=bottom_frame.winfo_reqheight(), borderwidth=0)
            imageBottomLabel.grid(row=0, column=0)

        #Buttons Tk
        # Button(NFC_Tk, text="Cerrar Ventana", command=lambda:closeTk()).pack(pady=10)

    def userManagementTk(self, user):
        # Config tk
        userManagement = Toplevel()
        userManagement.resizable(False,False)
        userManagement.title("Gestion de usuarios")
        # userManagement.overrideredirect(True)
        userManagement.geometry('800x600')
        userManagement.config(bg='#CCEEFF')

        # def Windows tk
        def createUserTk(userManagement):
            createUserWindows = Toplevel(userManagement)
            createUserWindows.resizable(False,False)
            createUserWindows.title("Gestion de usuarios")
            # updateUserWindows.overrideredirect(True)
            createUserWindows.geometry('800x600')
            createUserWindows.config(bg='#CCEEFF')

            usernameLabel = Label(createUserWindows, text='Nombre de usuario', bg='#CCEEFF')
            usernameLabel.grid()
            usernameEntry = Entry(createUserWindows)
            # # userEntry.bind("<1>", handle_click)
            usernameEntry.grid()

            nameLabel = Label(createUserWindows, text='Nombre', bg='#CCEEFF')
            nameLabel.grid()
            nameEntry = Entry(createUserWindows)
            nameEntry.grid()

            last_nameLabel = Label(createUserWindows, text='Apellido', bg='#CCEEFF')
            last_nameLabel.grid()
            last_nameEntry = Entry(createUserWindows)
            last_nameEntry.grid()

            emailLabel = Label(createUserWindows, text='Email', bg='#CCEEFF')
            emailLabel.grid()
            emailEntry = Entry(createUserWindows)
            emailEntry.grid()

            create = Button(createUserWindows, text='Crear', bg='#CCEEFF')
            create.grid()

            exitButton = Button(createUserWindows, text="Cerrar", command=lambda:exitTk(createUserWindows))
            exitButton.grid()
        
        def updateUserTk(userManagement):
            updateUserWindows = Toplevel(userManagement)
            updateUserWindows.resizable(False,False)
            updateUserWindows.title("Gestion de usuarios")
            # updateUserWindows.overrideredirect(True)
            updateUserWindows.geometry('800x600')
            updateUserWindows.config(bg='#CCEEFF')

            usernameLabel = Label(updateUserWindows, text='Nombre de usuario', bg='#CCEEFF')
            usernameLabel.grid()
            usernameEntry = Entry(updateUserWindows)
            # # userEntry.bind("<1>", handle_click)
            usernameEntry.grid()

            nameLabel = Label(updateUserWindows, text='Nombre', bg='#CCEEFF')
            nameLabel.grid()
            nameEntry = Entry(updateUserWindows)
            nameEntry.grid()

            last_nameLabel = Label(updateUserWindows, text='Apellido', bg='#CCEEFF')
            last_nameLabel.grid()
            last_nameEntry = Entry(updateUserWindows)
            last_nameEntry.grid()

            emailLabel = Label(updateUserWindows, text='Email', bg='#CCEEFF')
            emailLabel.grid()
            emailEntry = Entry(updateUserWindows)
            emailEntry.grid()

            create = Button(updateUserWindows, text='Modificar', bg='#CCEEFF')
            create.grid()

            exitButton = Button(updateUserWindows, text="Cerrar Sesion", command=lambda:exitTk(updateUserWindows))
            exitButton.grid()

        def deleteUserTk(userManagement):
            answerMessagebox = messagebox.askokcancel(title='Eliminar usuario', message='Desea eliminar el usuario')
            if answerMessagebox:
                print('Usuario eliminado')
            else:
                print('Acción cancelada')

        def logout(user):
            del user
            userManagement.destroy()

        def exitTk(windowsTk):
            windowsTk.destroy()
        
        # Frame Principal
        mainFrame = Frame(userManagement, width=800, height=600, bg='#CCEEFF')
        mainFrame.grid()

        # Create left and right frames
        left_frame = Frame(mainFrame, width=round(mainFrame.winfo_reqwidth()*0.5), height=round(mainFrame.winfo_reqheight()), bg='#CCEEFF')
        left_frame.grid(row=0, column=0)

        right_frame = Frame(mainFrame, width=round(mainFrame.winfo_reqwidth()*0.5), height=round(mainFrame.winfo_reqheight()), bg='#CCEEFF')
        right_frame.grid(row=0, column=1)

        # Buttons right_frame
        createUser = Button(right_frame, text='Crear', command=lambda:createUserTk(userManagement))
        createUser.grid()

        updateUser = Button(right_frame, text='Modificar/Actualizar', command=lambda:updateUserTk(userManagement))
        updateUser.grid()

        deleteUserButton = Button(right_frame, text='Bloquear/Eliminar', command=lambda:deleteUserTk(userManagement))
        deleteUserButton.grid()

        # ListBox
        langs = {'Java': 1, 'C#': 2, 'C': 3, 'C++': 4, 'Python': 5, 'Go': 6, 'JavaScript': 7, 'PHP' : 8, 'Swift': 9}
        listBox = Listbox(left_frame)
        listBox.grid()

        for key in langs:
            listBox.insert(END, '{}: {}'.format(key, langs[key]))


        exitButton = Button(right_frame, text="Cerrar Sesion", command=lambda:logout(user))
        exitButton.grid()

    def openConfigurationTk(self, user, adminConfigTk):
        # Config tk
        configurationTk = Toplevel()
        configurationTk.resizable(False,False)
        configurationTk.protocol("WM_DELETE_WINDOW", exit)
        configurationTk.title("Configuraciones")
        # configurationTk.overrideredirect(True)
        configurationTk.geometry('200x300')

        # width = 200
        # height = 300
        # screen_width = root.winfo_screenwidth()
        # screen_height = root.winfo_screenheight()

        # x = (screen_width/2) - (app_width/2)
        # y = (screen_height/2) - (app_height/2)

        # configurationTk.geometry(f'{width}x{height}+{int(x)}+{int(y)}')

        #Def
        def closeTk():
            configurationTk.destroy()
            # adminConfigTk.deiconify()

        def changeThreshold():
            while True:
                varThreshold = simpledialog.askstring(title="Threshold", prompt="Ingrese Threshold:")
                try:
                    varThreshold = float(varThreshold)
                    if varThreshold >=0 and varThreshold <= 1:
                        print (varThreshold)
                        return True
                    else:
                        print('Ingrese un numero valido entre 0 y 1 \n Ejemplo: 0.9')
                        messagebox.showinfo(title='Numero no valido', message='Ingrese un numero entre 0 y 1\nEjemplo: 0.9')
                except ValueError:
                    if not varThreshold == '':
                        messagebox.showerror(title='Caracter invalido', message='Solo admite numeros')
                except:
                    break

        def changeIouThreshold():
            while True:
                varIouThreshold = simpledialog.askstring(title="Iou Threshold", prompt="Ingrese Iou Threshold:")
                try:
                    varIouThreshold = float(varIouThreshold)
                    if varIouThreshold > 0 and varIouThreshold <= 1:
                        print(varIouThreshold)
                        return True
                    else:
                        print('Ingrese un numero valido entre 0 y 1 \n Ejemplo: 0.9')
                        messagebox.showinfo(title='Numero no valido', message='Ingrese un numero entre 0 y 1\nEjemplo: 0.9')
                except ValueError:
                    if not varIouThreshold == '':
                        messagebox.showerror(title='Caracter invalido', message='Solo admite numeros')
                except:
                    break

        def changeDetLimit():
            while True:
                varDetLimit = simpledialog.askstring(title="Limite de capturas", prompt="Ingrese limite de captura:")
                try:
                    varDetLimit = int(varDetLimit)
                    if varDetLimit >= 0 and varDetLimit <= 100:
                        print(varDetLimit)
                        return True
                    else:
                        print('Ingrese un numero valido entre 0 y 100 \n Ejemplo: 10')
                        messagebox.showinfo(title='Numero no valido', message='Ingrese un numero valido entre 0 y 100 \n Ejemplo: 10')
                except ValueError:
                    if not varDetLimit == '':
                        messagebox.showerror(title='Caracter invalido', message='Solo admite numeros')
                except:
                    break

        #Hide Root Window
        # root.withdraw()

        #Labels Tk
        # labelimagen = Label(configurationTk, image=imagen)
        # labelimagen.pack()

        labelTest = Label(configurationTk, text='fdgdf')
        labelTest.pack()

        labelThreshold = Label(configurationTk, text='Threshold : 5')
        labelThreshold.pack()

        labelIou_threshold = Label(configurationTk, text='Iou Threshold : 5')
        labelIou_threshold.pack()

        labelDetLimit = Label(configurationTk, text='Det limit : 5')
        labelDetLimit.pack()

        #Buttons Tk
        buttonDirectory = Button(configurationTk, text="Cambiar directorio", command=lambda:self.folderSelect())
        buttonDirectory.pack()

        buttonThreshold = Button(configurationTk, text="Cambiar Threshold", command=lambda:changeThreshold())
        buttonThreshold.pack()

        buttonIou_threshold = Button(configurationTk, text="Cambiar Iou Threshold", command=lambda:changeIouThreshold())
        buttonIou_threshold.pack()

        buttonDetLimit = Button(configurationTk, text="Cambiar limite de capturas", command=lambda:changeDetLimit())
        buttonDetLimit.pack()

        buttonClass = Button(configurationTk, text="Cambiar clases")
        buttonClass.pack()

        buttonClass = Button(configurationTk, text="Configurar Camaras", command=lambda:self.configCameraTk(configurationTk))
        buttonClass.pack()

        buttonfDirectory = Button(configurationTk, text="ImagenTest", command=lambda:self.folderframeSelect())
        buttonfDirectory.pack()

        closeWindow = Button(configurationTk, text="Cerrar Ventana", command=lambda:closeTk())
        closeWindow.pack()

    def adminConfigTk(self, user):
        adminConfigTk = Toplevel()
        adminConfigTk.title("Admin panel")
        adminConfigTk.resizable(False,False)
        adminConfigTk.config(background="#cceeff")
        adminConfigTk.resizable(False,False)
        # adminConfigTk.overrideredirect(True)
        # adminConfigTk.geometry(f'{root.winfo_screenwidth()}x{root.winfo_screenheight()}')
        adminConfigTk.geometry('300x300')

        def logout(user):
            del user
            adminConfigTk.destroy()

        #Labels

        usernameLabel = Label(adminConfigTk, text=user.getUsername())
        usernameLabel.grid()

        nameLabel = Label(adminConfigTk, text=user.getName())
        nameLabel.grid()

        Last_nameLabel = Label(adminConfigTk, text=user.getLast_name())
        Last_nameLabel.grid()

        # testButton = Button(adminConfigTk, text='Test download',command=self.downloadEfficientDet, fg='red').grid()
        testButton = Button(adminConfigTk, text='Test NFC',command=self.nfc_identifyTk, fg='red').grid()
        testButton = Button(adminConfigTk, text='Test POPUP',command=self.popupIdentificationTk, fg='red').grid()
        testButton = Button(adminConfigTk, text='test Cargar Dependencias',command=self.loadALL, fg='red').grid()

        createUser = Button(adminConfigTk, text='Gestion de usuario', command=lambda:self.userManagementTk(user))
        createUser.grid()

        configButton = Button(adminConfigTk, command=lambda:self.openConfigurationTk(user, adminConfigTk), text='Configuraciones')
        configButton.grid()

        exitButton = Button(adminConfigTk, text="Cerrar Sesion", command=lambda:logout(user))
        exitButton.grid()