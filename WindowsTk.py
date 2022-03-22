import queue
import time
from datetime import datetime, timedelta
from tkinter import *
from tkinter import messagebox, filedialog, simpledialog, Listbox, ttk
from PIL import ImageTk, Image
from mmpose.core import camera
import cv2
import pandas as pd
from numpy import empty
import torch.multiprocessing as mp
import torch

from Services import API_Services
from UserClass import Person
from imagenClipClass import imageClip
from FileManagementClass import FileManagement
from CameraStream import CameraStream
from PpeDetector import PpeDetector
from ActionDetector import ActionDetector
from NFCClass import NFC
from BTAudio import BTAudio




class WindowsTk:

    def __init__(self, root):
        self.root=root
        self.ppedet = PpeDetector()
        self.actiondet = ActionDetector()
        self.btaudio=BTAudio()
        # rtsp://admin:nvrHidrolatina@192.168.1.91:554/Streaming/channels/101
        # rtsp://admin:nvrHidrolatina@192.168.100.234:554/Streaming/channels/401
        # rtsp://admin:nvrHidrolatina@192.168.100.234:554/Streaming/channels/501

        self.detlimit=25
        self.varCamera='rtsp://admin:nvrHidrolatina@192.168.100.234:554/Streaming/channels/501'
        self.actiondet.varCamera='rtsp://admin:nvrHidrolatina@192.168.100.234:554/Streaming/channels/401'
    def center_window(self, window):
        window.update_idletasks()
        # get screen width and height
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        # calculate position x and y coordinates
        x = (screen_width/2) - (window.winfo_width()/2)
        y = (screen_height/2) - (window.winfo_height()/2)
        window.geometry('%dx%d+%d+%d' % (window.winfo_width(), window.winfo_height(), x, y))

    def loadALL(self):
        

        LoadTk = Toplevel()
        # LoadTk.resizable(False,False)
        # LoadTk.protocol("WM_DELETE_WINDOW", exit)
        LoadTk.title("Cargando") 
        # # LoadTk.config(bg='#CCEEFF')
        LoadTk.overrideredirect(True)
        image_bg = Image.open("images/ai_brain_logo.png")
        image_bg = image_bg.resize((int(image_bg.size[0]/4),int(image_bg.size[1]/4)))
        LoadTk.geometry(f'{image_bg.size[0]}x{image_bg.size[1]}')
        LoadTk.lift()
        LoadTk.attributes('-topmost', True)
        LoadTk.attributes('-topmost', False)  
        self.center_window(LoadTk)
        LoadTk.attributes('-transparentcolor', 'purple')
 
        
        mainFrame = Frame(LoadTk, bg='yellow')
        mainFrame.place(relx=0,rely=0, relwidth=1, relheight=1)

        image_bg = ImageTk.PhotoImage(image_bg)
        label_bg = Label(mainFrame, image=image_bg, bg='blue')
        label_bg.image=image_bg
        label_bg.place(relx=0,rely=0, relwidth=1, relheight=1)

        barFrame=Frame(mainFrame, bg='gray')
        barFrame.place(relx=0,rely=.95, relwidth=1, relheight=.05)

        stringbar = StringVar()
        stringbar.set("Cargando  0%")

        labelbar=Label(barFrame,textvariable=stringbar, bg='gray', font=('Digital-7',10))
        labelbar.place(relx=.1, rely=.5, anchor="center")
        labelbar.config(fg='white')

        s = ttk.Style()
        s.theme_use('clam')
        s.configure("bar.Horizontal.TProgressbar", bordercolor='black', background='lime', throughcolor='lime')

        progress_bar=ttk.Progressbar(barFrame, style="bar.Horizontal.TProgressbar", orient=HORIZONTAL,length=250 ,mode='determinate')
        progress_bar.place(relx=.85, rely=.5, relheight=.5, anchor="center")

        LoadTk.update()

        # try:
        #     del self.ppedet
        #     del self.actiondet
        #     self.ppedet
        #     self.actiondet
        # except AttributeError:
        #     self.ppedet = PpeDetector()
        #     self.actiondet = ActionDetector()
        
        self.queue_anno = mp.Queue()
        self.queue_action = mp.Queue()
        self.flag_posec3d_init=mp.Queue()
        self.p0 = mp.Process(target=self.actiondet.proc_paral, args=(self.queue_anno, self.queue_action, self.flag_posec3d_init,))
        self.p0.start()

        # self.queue_audio=mp.Queue()
        self.p1 = mp.Process(target=self.btaudio.playAudio, args=())
        self.p1.start()
    
        

        self.ppedet.loadEfficientDet()
        progress_bar['value']=15
        stringbar.set('Cargando 15% ')
        LoadTk.update_idletasks()
        LoadTk.update()

        self.ppedet.importMdetr.init()
        progress_bar['value']=30
        stringbar.set('Cargando 30%')
        LoadTk.update_idletasks()
        LoadTk.update()
        
        self.ppedet.loadClip()
        progress_bar['value']=45
        stringbar.set('Cargando 45%')
        LoadTk.update_idletasks()
        LoadTk.update()

        self.actiondet.load_effdet()
        progress_bar['value']=60
        stringbar.set('Cargando 60%')
        LoadTk.update_idletasks()
        LoadTk.update()
        self.actiondet.load_pose()
        progress_bar['value']=70
        stringbar.set('Cargando 70%')
        LoadTk.update_idletasks()
        LoadTk.update()
        self.actiondet.load_zone()
        progress_bar['value']=85
        stringbar.set('Cargando 85%')
        LoadTk.update_idletasks()
        LoadTk.update()
        while self.flag_posec3d_init.empty():
            pass
        progress_bar['value']=100
        # labelbar.config(fg='green')
        stringbar.set('Carga Completa 100%')
        LoadTk.update_idletasks()
        LoadTk.update()
        torch.cuda.empty_cache()
        # messagebox.showinfo(message="Dependencias cargadas")
        LoadTk.after(2100, LoadTk.destroy)

    ########Windows#######

    #Def
    def popup(self, message):
        messagebox.showinfo(message=message)

    def folderSelect(self):
        folder_selected = filedialog.askdirectory()
        print(folder_selected)

    def folderPpeframeSelect(self):
        # global ppeframe_selected
        self.ppeframe_selected = filedialog.askopenfilename()
        if not self.ppeframe_selected == '':
            try:
                del self.ppevideo_selected
            except: 
                pass

            print('PPE Image: ', self.ppeframe_selected)
            
            self.ppeimageLabel.config(text='{}'.format(self.ppeframe_selected))
            self.ppevideoLabel.config(text='No')
        else:
            del self.ppeframe_selected
            self.ppeimageLabel.config(text='No')

    def folderPpevideoSelect(self):
        # global ppeframe_selected
        self.ppevideo_selected = filedialog.askopenfilename()
        if not self.ppevideo_selected=='':
            print(self.ppevideo_selected)
            try:
                del self.ppeframe_selected
            except: 
                pass
            self.ppevideoLabel.config(text='{}'.format(self.ppevideo_selected))
            self.ppeimageLabel.config(text='No')
        else:
            del self.ppevideo_selected
            self.ppevideoLabel.config(text='No')
    def folderactionvideoSelect(self):
        # global ppeframe_selected
        self.actiondet.actionvideo_selected = filedialog.askopenfilename()
        if not self.actiondet.actionvideo_selected=='':
            print('Action Image: ', self.actiondet.actionvideo_selected)
            self.actionvideoLabel.config(text='{}'.format(self.actiondet.actionvideo_selected ))
        else:
            del self.actiondet.actionvideo_selected
            self.actionvideoLabel.config(text='No')
    ###################Def Windows's###################

    def showPytorchCameraTk(self, user, hide=False):
        self.btaudio.play('Bienvenido operador, por favor espera unos segundos frente a la pantalla, gracias!')
        # import datetime
        import numpy as np

        #Var/Global
        # global det
        # global image
        # global original_image
        self.det=0
        #Tkinter config
        self.PytorchCameraTk = Toplevel()
        # if hide:
        #     self.PytorchCameraTk.withdraw()
        self.PytorchCameraTk.title('Camara')
        # self.PytorchCameraTk.resizable(False,False)
        self.PytorchCameraTk.overrideredirect(True)
        self.PytorchCameraTk.geometry(f'{self.PytorchCameraTk.winfo_screenwidth()}x{self.PytorchCameraTk.winfo_screenheight()}')
        self.PytorchCameraTk.config(background="red")
        self.center_window(self.PytorchCameraTk)

        # self.PytorchCameraTk.geometry(f'{1280}x{720}')
    
        # self.PytorchCameraTk.geometry("1280x720")

        # self.image = PhotoImage(file="white-image.png")
        # self.original_image = self.image.subsample(1,1)

        mainFrame = Frame(self.PytorchCameraTk, width=self.PytorchCameraTk.winfo_screenwidth(), height=self.PytorchCameraTk.winfo_screenheight(), bg="#cceeff", borderwidth=0)
        mainFrame.place(x=0,y=0)
        # mainFrame.grid()
        # mainFrame.grid_propagate(False)
        
        self.PytorchCameraTk.update_idletasks()

        canvas = Canvas(mainFrame, borderwidth=0,highlightthickness=0)
        canvas.place(relx=.5, rely=.5, relwidth=1, relheight=1, anchor='center')

        ##BackGround Image
        bg = Image.open('images/bg.jpg')
        bg = bg.resize((mainFrame.winfo_screenwidth(), mainFrame.winfo_screenheight()), Image.ANTIALIAS)
        bg = ImageTk.PhotoImage(bg)
        self.PytorchCameraTk.bg=bg
        canvas.create_image(mainFrame.winfo_screenwidth()/2, mainFrame.winfo_screenheight()/2, image=bg)

        #Logo
        logo_icon= Image.open('images/logo_hidrolatina_h.png')
        logo_icon= logo_icon.resize((int(mainFrame.winfo_screenwidth()*.025), int(mainFrame.winfo_screenheight()*.05)), Image.ANTIALIAS)
        logo_icon= ImageTk.PhotoImage(logo_icon)
        self.PytorchCameraTk.logo_icon=logo_icon
        canvas.create_image(int(mainFrame.winfo_screenwidth()*.95), int(mainFrame.winfo_screenheight()*.05), image=logo_icon)

        exitImg = Image.open('images/backButton.png')
        exitImg = exitImg.resize((int(mainFrame.winfo_height()*.05), int(mainFrame.winfo_height()*.05)), Image.ANTIALIAS)
        exitImg = ImageTk.PhotoImage(exitImg)
        self.PytorchCameraTk.exitImg=exitImg
        exitCanvas=canvas.create_image(mainFrame.winfo_width()*.05, mainFrame.winfo_height()*.05, image=exitImg)
        canvas.tag_bind(exitCanvas, "<Button-1>",  (lambda _:closeTk()))

        self.PytorchCameraTk.update_idletasks()
        # ###Label imageHeadFrame Sub frame lvl 2 headFrame
        headImg=Label(mainFrame,  borderwidth=2, relief="sunken")
        headImg.place(relx=.7, rely=.15, width=int(mainFrame.winfo_height()*.2), height=int(mainFrame.winfo_height()*.2))

        # ###Label imageHandFrame Sub frame lvl 2 handFrame
        handImg=Label(mainFrame, borderwidth=2, relief="sunken")
        handImg.place( relx=.7, rely=.4, width=int(mainFrame.winfo_height()*.2), height=int(mainFrame.winfo_height()*.2))
   

        # ###Label imagebootFrame Sub frame lvl 2 bootFrame
        bootImg=Label(mainFrame, borderwidth=2, relief="sunken")
        bootImg.place(relx=.7, rely=.65, width=int(mainFrame.winfo_height()*.2), height=int(mainFrame.winfo_height()*.2))
    

        # ####Label dataHeadFrame Sub frame lvl 2 headFrame
        Label(mainFrame, text="Casco",font=('Digital-7',15), relief='sunken', borderwidth=1).place(relx=.83, rely=.15, relwidth=0.05)
        helmetLabel=Label(mainFrame,font=('Digital-7',15), relief='sunken', borderwidth=1)
        helmetLabel.place(relx=.89, rely=.15, relwidth=0.09)

        Label(mainFrame, text="Audífonos",font=('Digital-7',14), relief='sunken', borderwidth=1).place(relx=.83, rely=.20, relwidth=0.05)
        headphonesLabel=Label(mainFrame,font=('Digital-7',14), relief='sunken', borderwidth=1)
        headphonesLabel.place(relx=.89, rely=.20, relwidth=0.09)

        Label(mainFrame, text="Antiparras",font=('Digital-7',14), relief='sunken', borderwidth=1).place(relx=.83, rely=.25, relwidth=0.05)
        gogglesLabel=Label(mainFrame,font=('Digital-7',14), relief='sunken', borderwidth=1)
        gogglesLabel.place(relx=.89, rely=.25, relwidth=0.09)

        Label(mainFrame, text="Mascarilla",font=('Digital-7',14), relief='sunken', borderwidth=1).place(relx=.83, rely=.30, relwidth=0.05)
        maskLabel=Label(mainFrame,font=('Digital-7',14), relief='sunken', borderwidth=1)
        maskLabel.place(relx=.89, rely=.30, relwidth=0.09)

        # ####Label dataHandFrame Sub frame lvl 2 handFrame
        Label(mainFrame, text="Guantes",font=('Digital-7',14), relief='sunken', borderwidth=1).place(relx=.83, rely=.4, relwidth=0.05)
        glovesLabel=Label(mainFrame,font=('Digital-7',14), relief='sunken', borderwidth=1)
        glovesLabel.place(relx=.89, rely=.4, relwidth=0.09)

        # ####Label dataBootFrame Sub frame lvl 2 bootFrame
        Label(mainFrame, text="Botas",font=('Digital-7',14), relief='sunken', borderwidth=1).place(relx=.83, rely=.65, relwidth=0.05)
        bootsLabel=Label(mainFrame,font=('Digital-7',14), relief='sunken', borderwidth=1)
        bootsLabel.place(relx=.89, rely=.65, relwidth=0.09)

        #Capture video frames
        labelVideo = Label(mainFrame, bg="#cceeff", borderwidth=3, relief="sunken")
        labelVideo.place(relx=.05, rely=.15, relwidth=.6, relheight=.7)

        if hide:
            self.PytorchCameraTk.withdraw()
        
        print('Camara de epp es esta: ', self.varCamera)  

        # try:
        #     self.cap.stop()
        # except:
        #     pass
        # try:
        #     del self.cap
        # except:
        #     pass

        try:
            self.ppeframe_selected
        except AttributeError:
            try:
                self.cap = CameraStream(self.ppevideo_selected,delay=0.03).start()
            except:
                self.cap = CameraStream(self.varCamera).start()
                print('COMENZO LA CAMARA')
        
        # self.cap = cv2.VideoCapture(0)

        # camWidth = round(cameraFrame.winfo_reqwidth())
        # camHeight = round(cameraFrame.winfo_reqheight()*0.85)

        #Def into tk
        def closeTk():
            #Destroy window
            try:
                self.cap.stop()
            except:
                pass
            self.PytorchCameraTk.destroy()
            # root.deiconify()
        
        def showFrame():
            # _, frame = self.cap.read()
            # frame = cv2.flip(frame, 1)
            try:
                frame=cv2.imread(self.ppeframe_selected)
            except:
                frame = self.cap.read()

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

            cv2image = cv2.cvtColor(cv2.resize(ori_img, (labelVideo.winfo_width(), labelVideo.winfo_height())), cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            labelVideo.imgtk = imgtk
            labelVideo.configure(image=imgtk)

            print(self.det)
            if len(out['class_ids']) == 0 or len(out['class_ids'])>1:
                self.det = 0 
            if len(out['class_ids']) == 1:
                self.det += 1
                if self.det==self.detlimit:

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
                    # self.cap.stop()
                    copy_imgtk = imgtk
                    labelVideo.imgtk = copy_imgtk

                    mdetr_list=self.ppedet.MDETR(im)
                    print(mdetr_list)
                    self.listImagenClip = []
                    for bodypart in mdetr_list.keys(): 
                        self.listImagenClip.append(imageClip(self.ppedet.names_ppe[bodypart], ImageTk.PhotoImage(mdetr_list[bodypart].resize((int(mainFrame.winfo_height()*.2),int(mainFrame.winfo_height()*.2)))), self.ppedet.clip(bodypart, mdetr_list)))
                    
                    updateLabel()
                    # return self
            
            if self.det<self.detlimit:
                labelVideo.after(10, showFrame)
        
        def counterPopUp(endTime, booleanAnswerlist):
            if datetime.now() > endTime:
                print('si')
                print(datetime.now().strftime('%H:%M:%S'), endTime.strftime('%H:%M:%S'))
                print('funciona')
                self.PytorchCameraTk.after(1500, closeTk)
                self.popupIdentificationTk(booleanAnswerlist)
                
            else:
                print('no')
                print(datetime.now().strftime('%H:%M:%S'), endTime.strftime('%H:%M:%S'))
                self.PytorchCameraTk.after(5000, counterPopUp, endTime, booleanAnswerlist)

        def updateLabel():
            # self.PytorchCameraTk.update_idletasks()

            #Head Frame
            headImg.config(image=(self.listImagenClip[0].getImage()))
            helmetLabel.config(text=(self.listImagenClip[0].getAnswer()[0]))
            if self.listImagenClip[0].getAnswer()[0]=='Ok':
                helmetLabel.config(bg='lime')
            else:
                helmetLabel.config(bg='#ff4040')
            headphonesLabel.config(text=(self.listImagenClip[0].getAnswer()[1]))
            if self.listImagenClip[0].getAnswer()[1]=='Ok':
                headphonesLabel.config(bg='lime')
            else:
                headphonesLabel.config(bg='#ff4040')

            gogglesLabel.config(text=(self.listImagenClip[0].getAnswer()[2]))
            if self.listImagenClip[0].getAnswer()[2]=='Ok':
                gogglesLabel.config(bg='lime')
            else:
                gogglesLabel.config(bg='#ff4040')

            maskLabel.config(text=(self.listImagenClip[0].getAnswer()[3]))
            if self.listImagenClip[0].getAnswer()[3]=='Ok':
                maskLabel.config(bg='lime')
            else:
                maskLabel.config(bg='#ff4040')
            #Hand Frame
            handImg.config(image=(self.listImagenClip[1].getImage()))      
            glovesLabel.config(text=(self.listImagenClip[1].getAnswer()[0]))
            if self.listImagenClip[1].getAnswer()[0]=='Ok':
                glovesLabel.config(bg='lime')
            else:
                glovesLabel.config(bg='#ff4040')

            #Boot Frame
            bootImg.config(image=(self.listImagenClip[2].getImage()))
            bootsLabel.config(text=(self.listImagenClip[2].getAnswer()[0]))
            if self.listImagenClip[2].getAnswer()[0]=='Ok':
                bootsLabel.config(bg='lime')
            else:
                bootsLabel.config(bg='#ff4040')

            # self.cap.stop()

            booleanAnswer = None
            booleanAnswerlist=[]
            for list in self.listImagenClip:
                for j in range(len(list.getAnswer())):
                    if list.getAnswer()[j] == 'Ok':
                        print(list.getName()[j])
                        print(list.getAnswer()[j])
                        booleanAnswer = True
                        booleanAnswerlist.append(booleanAnswer)
                    else:
                        booleanAnswer = False
                        booleanAnswerlist.append(booleanAnswer)
                    print(booleanAnswer)

            endTime = datetime.now() + timedelta(seconds=5)
            print('PRUEBA DE ERROR ')
            if len(self.listImagenClip) > 0:
                counterPopUp(endTime, booleanAnswerlist)

        if hide:
            self.PytorchCameraTk.withdraw()
            self.PytorchCameraTk.after(3000,self.PytorchCameraTk.deiconify())
        try:
            showFrame()
        except TypeError:
            print('Error frame vacío')
            pass
        self.PytorchCameraTk.focus_force()
        # self.PytorchCameraTk.withdraw()
        # self.PytorchCameraTk.after(2000,self.PytorchCameraTk.deiconify)
        self.PytorchCameraTk.lift()
        self.PytorchCameraTk.attributes('-topmost', True)
        self.PytorchCameraTk.after_idle(self.PytorchCameraTk.attributes,'-topmost',False)

    def showActionsTk(self):
        showActions = Toplevel()
        # showActions.update() 
        showActions.resizable(False,False)
        showActions.title("Configuracion camaras")
        # showActions.protocol("WM_DELETE_WINDOW", exit)
        showActions.config(background="#cceeff")
        showActions.overrideredirect(True)
        # showActions.geometry('1000x600')
        showActions.geometry(f'{showActions.winfo_screenwidth()}x{showActions.winfo_screenheight()}')
        self.center_window(showActions)
        
        showActions.focus_force()
        showActions.lift()
        showActions.attributes('-topmost', True)
        showActions.after_idle(showActions.attributes,'-topmost',False)

        #Def into tk
        def closeTk():
            try:
                self.actiondet.cam.stop()
            except:
                pass
            showActions.destroy()
            # root.deiconify()
        def DownloadpdTk():
            out = filedialog.asksaveasfilename(defaultextension=".xlsx")
            print('out ', out)
            self.actiondet.df.to_excel(out, index=False)
            # print(self.df)
           
        
        # Main Frame
        mainFrame = Frame(showActions, bg='#cceeff', borderwidth=0)
        mainFrame.place(x=0, y=0, relwidth=1, relheight=1)


        showActions.update_idletasks()

        canvas = Canvas(mainFrame, borderwidth=0,highlightthickness=0, bg='blue')
        canvas.place(relx=.5, rely=.5, relwidth=1, relheight=1, anchor='center')

        bg = Image.open('images/bg.jpg')
        bg = bg.resize((showActions.winfo_screenwidth(), showActions.winfo_screenheight()), Image.ANTIALIAS)
        bg = ImageTk.PhotoImage(bg)
        showActions.bg=bg
        canvas.create_image(showActions.winfo_screenwidth()/2, showActions.winfo_screenheight()/2, image=bg)

        #Logo
        logo_icon= Image.open('images/logo_hidrolatina_h.png')
        logo_icon= logo_icon.resize((int(mainFrame.winfo_screenheight()*.05), int(mainFrame.winfo_screenheight()*.05)), Image.ANTIALIAS)
        logo_icon= ImageTk.PhotoImage(logo_icon)
        showActions.logo_icon=logo_icon
        canvas.create_image(int(mainFrame.winfo_screenwidth()*.95), int(mainFrame.winfo_screenheight()*.05), image=logo_icon)


        #Buttons
        downloadWindow= Button(mainFrame, text="Descargar Historial", command=DownloadpdTk)
        downloadWindow.place(relx=.75, rely=.75)


        exitImg = Image.open('images/backButton.png')
        exitImg = exitImg.resize((int(showActions.winfo_screenheight()*.05), int(showActions.winfo_screenheight()*.05)), Image.ANTIALIAS)
        exitImg = ImageTk.PhotoImage(exitImg)
        showActions.exitImg=exitImg

        blank=canvas.create_image(showActions.winfo_screenwidth()*.05, showActions.winfo_screenheight()*.05, image=exitImg)
        exitButton = canvas.tag_bind(blank, "<Button-1>",  (lambda _:closeTk()))

        #Capture video frames
        labelVideo = Label(mainFrame, relief='sunken', borderwidth=3)
        labelVideo.place(relx=.55, rely=0.175,relwidth=.4, relheight=.4)


        labelData = LabelFrame(mainFrame)
        labelData.place(relx=.05, rely=0.175,relwidth=.45, relheight=.5)
        # labelData.grid_propagate(False)

        tv1 = ttk.Treeview(labelData)
        tv1.place(relheight=1, relwidth=1) # set the height and width of the widget to 100% of its container (frame1).
        # tv1.place(relx=1, rely=1, anchor="c")        
        
        treescrolly = Scrollbar(labelData, orient="vertical", command=tv1.yview) # command means update the yaxis view of the widget
        treescrollx = Scrollbar(labelData, orient="horizontal", command=tv1.xview) # command means update the xaxis view of the widget
        tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set) # assign the scrollbars to the Treeview Widget
        treescrollx.pack(side=BOTTOM, fill=X) # make the scrollbar fill the x axis of the Treeview widget
        treescrolly.pack(side=RIGHT, fill=Y)


        showActions.focus_force()
        showActions.lift()
        showActions.attributes('-topmost', True)
        showActions.after_idle(showActions.attributes,'-topmost',False)
        self.actiondet.inferenceActionDetector(self.queue_anno, self.queue_action, labelVideo, showActions, tv1, self.btaudio)
        
    
        
    def configCameraPPETk(self):
        # Config tk
      
        varCamera = simpledialog.askstring(title="Camara", prompt="Ingrese ID o URL de Camara PPE:")
        if varCamera == '' or varCamera==None:
            # self.varCamera = 0
            print('Camara EPP: ',self.varCamera)
        else:
            if varCamera.isdigit():
                varCamera=int(varCamera)
            print('Camara EPP: ',self.varCamera)
            self.varCamera=varCamera
            self.cameraLabel.config(text='Camara PPE: {}           Camara Planta: {}'.format(self.varCamera, self.actiondet.varCamera))
    def configCameraActionTk(self):
        # Config tk
      
        varCamera = simpledialog.askstring(title="Camara", prompt="Ingrese ID o URL de Camara Planta:")
        if varCamera == '' or varCamera==None:
            # self.varCamera = 0
            print('Camara Planta: ', self.actiondet.varCamera)
        else:
            if varCamera.isdigit():
                varCamera=int(varCamera)
            print('Camara Planta: ', self.actiondet.varCamera)
            self.actiondet.varCamera=varCamera
            self.cameraLabel.config(text='Camara PPE: {}           Camara Planta: {}'.format(self.varCamera, self.actiondet.varCamera))

 
    def nfc_identifyTk(self):
        # import concurrent.futures
        # Config tk
        self.NFC_Tk = Toplevel()
        # self.NFC_Tk.resizable(False,False)
        self.NFC_Tk.title("Identificación")
        self.NFC_Tk.overrideredirect(True)
        self.NFC_Tk.geometry(f'{self.NFC_Tk.winfo_screenwidth()}x{self.NFC_Tk.winfo_screenheight()}')

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
            self.NFC_Tk.destroy()
            # root.deiconify()
        
        #Hide Root Window
        # root.withdraw()

        # Frame
        NFCFrame = Frame(self.NFC_Tk, width=self.NFC_Tk.winfo_screenwidth(), height=self.NFC_Tk.winfo_screenheight(), bg='#CCEEFF')
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
        self.NFC_Tk.focus_force()
        # self.NFC_Tk.lift()
        # self.NFC_Tk.attributes('-topmost', True)
        # self.NFC_Tk.after_idle(self.NFC_Tk.attributes,'-topmost',False)

        # thread= Thread(target=identify, args=())
        # thread.start()
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #         future = executor.submit(NFC.identify)
                # return_value = future.result()
                # print(return_value)

        # self.NFC_Tk.after(3000, thread_identify)

        #Buttons Tk
        # Button(self.NFC_Tk, text="Cerrar Ventana", command=lambda:closeTk()).pack(pady=10)

        # thread= Thread(target=NFC.identify, args=())
        # thread.start()

        # thread.join()

        return self.NFC_Tk

    def popupIdentificationTk(self, booleanAnswerlist):
        # Config tk
        popupIdentificationTk = Toplevel()
        popupIdentificationTk.resizable(False,False)
        popupIdentificationTk.after(6000, popupIdentificationTk.destroy)
        popupIdentificationTk.overrideredirect(True)
        popupIdentificationTk.geometry(f'{popupIdentificationTk.winfo_screenwidth()}x{popupIdentificationTk.winfo_screenheight()}')
        # popupIdentificationTk.geometry("1280x720")

        # def closeTk():
        #     #Destroy window
        #     try:
        #         self.cap.stop()
        #     except:
        #         pass
        #     self.PytorchCameraTk.destroy()

        #Code
        def no_ppe(boollist):
            namelist=[]
            no_ppe_text='Te faltan los siguientes EPP:\n\n'
            no_ppe_audio=''
            n=1
            for parts in self.ppedet.names_ppe:
                for elem in self.ppedet.names_ppe[parts]:
                    namelist.append(elem)
            print('namelist: ', namelist)        
            for i in enumerate(namelist):
                if not boollist[i[0]]:
                    no_ppe_text+=('    '+str(n)+'. '+i[1]+'\n')
                    no_ppe_audio+=f'{i[1]}, '
                    n+=1
            no_ppe_audio=no_ppe_audio[:-2]
            no_ppe_audio= no_ppe_audio[:no_ppe_audio.rfind(',')]+' y '+ no_ppe_audio[no_ppe_audio.rfind(',')+2:]
            print(no_ppe_text)        
            return no_ppe_text, no_ppe_audio

        PopUpIdentificationFrame = Frame(popupIdentificationTk, width=popupIdentificationTk.winfo_screenwidth(), height=popupIdentificationTk.winfo_screenheight(), bg='#CCEEFF')
        PopUpIdentificationFrame.grid()

        if all(elem == True for elem in booleanAnswerlist):
            global detections
            self.btaudio.play('Tienes todos tus elementos de seguridad, pasa a tu zona de trabajo')
            detections = Image.open('images/approved_detections.png')
            detections = detections.resize((PopUpIdentificationFrame.winfo_reqwidth(), PopUpIdentificationFrame.winfo_reqheight()), Image.ANTIALIAS)
            detections = ImageTk.PhotoImage(detections)

            imageFrame = Frame(PopUpIdentificationFrame, width=PopUpIdentificationFrame.winfo_reqwidth(), height=PopUpIdentificationFrame.winfo_reqheight())
            imageFrame.grid(row=0, column=0)
            imageLabel = Label(imageFrame, image=detections)
            imageLabel.pack()
            # popupIdentificationTk.after(4000, self.PytorchCameraTk.destroy)
            # popupIdentificationTk.after(4000, closeTk)
            popupIdentificationTk.after(3000, self.showActionsTk)


        else:
            global imageTop, imageMiddleLeft, imageMiddleRight, imageBottom
            ppelist=no_ppe(booleanAnswerlist)

            self.btaudio.play(f'Por favor colócate tu {ppelist[1]}')
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

            imageMiddleLeftLabel = Label(left_middle_frame, text=ppelist[0],font='Digital-7 32 bold', borderwidth=0, bg='#CCEEFF', justify='left', anchor='w')
            imageMiddleLeftLabel.place(relx=.15, rely=0, relwidth=.85, relheight=1)

            imageMiddleRightLabel = Label(right_middle_frame, image=imageMiddleRight, width=right_middle_frame.winfo_reqwidth(), height=right_middle_frame.winfo_reqheight(), borderwidth=0)
            imageMiddleRightLabel.grid(row=0, column=0)

            imageBottomLabel = Label(bottom_frame, image=imageBottom, width=bottom_frame.winfo_reqwidth(), height=bottom_frame.winfo_reqheight(), borderwidth=0)
            imageBottomLabel.grid(row=0, column=0)

            # popupIdentificationTk.after(3500, self.PytorchCameraTk.destroy)
            # popupIdentificationTk.after(3500, closeTk)
            popupIdentificationTk.after(5000, lambda: NFC(self.nfc_identifyTk, self.showPytorchCameraTk))
            
        #Buttons Tk
        # Button(self.NFC_Tk, text="Cerrar Ventana", command=lambda:closeTk()).pack(pady=10)
        popupIdentificationTk.focus_force()
        popupIdentificationTk.lift()
        popupIdentificationTk.attributes('-topmost', True)
        popupIdentificationTk.after_idle(popupIdentificationTk.attributes,'-topmost',False)

    def userManagementTk(self, user):
        # Config tk
        userManagement = Toplevel()
        userManagement.title("Gestion de usuarios")
        userManagement.resizable(False,False)
        userManagement.config(background="#cceeff")
        # adminConfigTk.resizable(False,False)
        userManagement.overrideredirect(True)
        userManagement.geometry(f'{userManagement.winfo_screenwidth()}x{userManagement.winfo_screenheight()}')
        self.center_window(userManagement)

        #Canvas
        canvas = Canvas(userManagement, borderwidth=0,highlightthickness=0)
        canvas.place(relx=.5, rely=.5, relwidth=1, relheight=1, anchor='center')

        bg = Image.open('images/network_bg.png')
        bg = bg.resize((userManagement.winfo_screenwidth(), userManagement.winfo_screenheight()), Image.ANTIALIAS)
        bg = ImageTk.PhotoImage(bg)
        userManagement.bg=bg
        canvas.create_image(userManagement.winfo_screenwidth()/2, userManagement.winfo_screenheight()/2, image=bg)

        # # def Windows tk
        # def createUserTk(userManagement):
        #     createUserWindows = Toplevel(userManagement)
        #     createUserWindows.resizable(False,False)
        #     createUserWindows.title("Gestion de usuarios")
        #     # updateUserWindows.overrideredirect(True)
        #     createUserWindows.geometry('800x600')
        #     createUserWindows.config(bg='#CCEEFF')

        #     usernameLabel = Label(createUserWindows, text='Nombre de usuario', bg='#CCEEFF')
        #     usernameLabel.grid()
        #     usernameEntry = Entry(createUserWindows)
        #     # # userEntry.bind("<1>", handle_click)
        #     usernameEntry.grid()

        #     nameLabel = Label(createUserWindows, text='Nombre', bg='#CCEEFF')
        #     nameLabel.grid()
        #     nameEntry = Entry(createUserWindows)
        #     nameEntry.grid()

        #     last_nameLabel = Label(createUserWindows, text='Apellido', bg='#CCEEFF')
        #     last_nameLabel.grid()
        #     last_nameEntry = Entry(createUserWindows)
        #     last_nameEntry.grid()

        #     emailLabel = Label(createUserWindows, text='Email', bg='#CCEEFF')
        #     emailLabel.grid()
        #     emailEntry = Entry(createUserWindows)
        #     emailEntry.grid()

        #     create = Button(createUserWindows, text='Crear', bg='#CCEEFF')
        #     create.grid()

        #     exitButton = Button(createUserWindows, text="Cerrar", command=lambda:exitTk(createUserWindows))
        #     exitButton.grid()
        
        # def updateUserTk(userManagement):
        #     updateUserWindows = Toplevel(userManagement)
        #     updateUserWindows.resizable(False,False)
        #     updateUserWindows.title("Gestion de usuarios")
        #     # updateUserWindows.overrideredirect(True)
        #     updateUserWindows.geometry('800x600')
        #     updateUserWindows.config(bg='#CCEEFF')

        #     usernameLabel = Label(updateUserWindows, text='Nombre de usuario', bg='#CCEEFF')
        #     usernameLabel.grid()
        #     usernameEntry = Entry(updateUserWindows)
        #     # # userEntry.bind("<1>", handle_click)
        #     usernameEntry.grid()

        #     nameLabel = Label(updateUserWindows, text='Nombre', bg='#CCEEFF')
        #     nameLabel.grid()
        #     nameEntry = Entry(updateUserWindows)
        #     nameEntry.grid()

        #     last_nameLabel = Label(updateUserWindows, text='Apellido', bg='#CCEEFF')
        #     last_nameLabel.grid()
        #     last_nameEntry = Entry(updateUserWindows)
        #     last_nameEntry.grid()

        #     emailLabel = Label(updateUserWindows, text='Email', bg='#CCEEFF')
        #     emailLabel.grid()
        #     emailEntry = Entry(updateUserWindows)
        #     emailEntry.grid()

        #     create = Button(updateUserWindows, text='Modificar', bg='#CCEEFF')
        #     create.grid()

        #     exitButton = Button(updateUserWindows, text="Cerrar Sesion", command=lambda:exitTk(updateUserWindows))
        #     exitButton.grid()

        # def deleteUserTk(userManagement):
        #     answerMessagebox = messagebox.askokcancel(title='Eliminar usuario', message='Desea eliminar el usuario')
        #     if answerMessagebox:
        #         print('Usuario eliminado')
        #     else:
        #         print('Acción cancelada')

        # def logout(user):
        #     del user
        #     userManagement.destroy()

        # def exitTk(windowsTk):
        #     windowsTk.destroy()
        
        # # Frame Principal
        # mainFrame = Frame(userManagement, width=800, height=600, bg='#CCEEFF')
        # mainFrame.grid()

        # # Create left and right frames
        # left_frame = Frame(mainFrame, width=round(mainFrame.winfo_reqwidth()*0.5), height=round(mainFrame.winfo_reqheight()), bg='#CCEEFF')
        # left_frame.grid(row=0, column=0)

        # right_frame = Frame(mainFrame, width=round(mainFrame.winfo_reqwidth()*0.5), height=round(mainFrame.winfo_reqheight()), bg='#CCEEFF')
        # right_frame.grid(row=0, column=1)

        # # Buttons right_frame
        # createUser = Button(right_frame, text='Crear', command=lambda:createUserTk(userManagement))
        # createUser.grid()

        # updateUser = Button(right_frame, text='Modificar/Actualizar', command=lambda:updateUserTk(userManagement))
        # updateUser.grid()

        # deleteUserButton = Button(right_frame, text='Bloquear/Eliminar', command=lambda:deleteUserTk(userManagement))
        # deleteUserButton.grid()

        # # ListBox
        # langs = {'Java': 1, 'C#': 2, 'C': 3, 'C++': 4, 'Python': 5, 'Go': 6, 'JavaScript': 7, 'PHP' : 8, 'Swift': 9}
        # listBox = Listbox(left_frame)
        # listBox.grid()

        # for key in langs:
        #     listBox.insert(END, '{}: {}'.format(key, langs[key]))


        # exitButton = Button(right_frame, text="Cerrar Sesion", command=lambda:logout(user))
        # exitButton.grid()

    def openConfigurationTk(self, user, adminConfigTk):
        # Config tk
        self.configurationTk = Toplevel()
        self.configurationTk.resizable(False,False)
        # self.configurationTk.protocol("WM_DELETE_WINDOW", exit)
        self.configurationTk.title("Configuraciones")
        self.configurationTk.overrideredirect(True)
        # self.configurationTk.geometry('200x400')
        self.configurationTk.geometry(f'{self.configurationTk.winfo_screenwidth()}x{self.configurationTk.winfo_screenheight()}')
        self.center_window(self.configurationTk)

        # width = 200
        # height = 300
        # screen_width = root.winfo_screenwidth()
        # screen_height = root.winfo_screenheight()

        # x = (screen_width/2) - (app_width/2)
        # y = (screen_height/2) - (app_height/2)

        # self.configurationTk.geometry(f'{width}x{height}+{int(x)}+{int(y)}')

        #Def
        def closeTk():
            adminConfigTk.focus_force()
            adminConfigTk.deiconify()
            self.configurationTk.destroy()
            

        def changeDetLimit():
            while True:
                varDetLimit = simpledialog.askstring(title="Limite de capturas", prompt="Ingrese limite de captura:")
                try:
                    varDetLimit = int(varDetLimit)
                    if varDetLimit >= 0 and varDetLimit <= 100:
                        print(varDetLimit)
                        self.detlimit=varDetLimit
                        self.detlimitLabel.config(text='Value: {}'.format(self.detlimit))
                        return True
                    else:
                        print('Ingrese un numero valido entre 0 y 100 \n Ejemplo: 10')
                        messagebox.showinfo(title='Numero no valido', message='Ingrese un numero valido entre 0 y 100 \n Ejemplo: 10')
                except ValueError:
                    if not varDetLimit == '':
                        messagebox.showerror(title='Caracter invalido', message='Solo admite numeros')
                except:
                    break
        
        def ppeimagetext():
            try:
                return self.ppeframe_selected
            except:
                return 'No'
        def ppevideotext():
            try:
                return self.ppevideo_selected
            except:
                return 'No'
        def actionvideotext():
            try:
                return self.actiondet.actionvideo_selected
            except:
                return 'No'
        #Hide Root Window
        # root.withdraw()

        canvas = Canvas(self.configurationTk, borderwidth=0,highlightthickness=0, bg='blue')
        canvas.place(relx=.5, rely=.5, relwidth=1, relheight=1, anchor='center')

        bg = Image.open('images/network_bg.png')
        bg = bg.resize((self.configurationTk.winfo_screenwidth(), self.configurationTk.winfo_screenheight()), Image.ANTIALIAS)
        bg = ImageTk.PhotoImage(bg)
        self.configurationTk.bg=bg
        canvas.create_image(self.configurationTk.winfo_screenwidth()/2, self.configurationTk.winfo_screenheight()/2, image=bg)

        logo = Image.open('images/logo_hidrolatina.png')
        logo = logo.resize((325, 97), Image.ANTIALIAS)
        logo = ImageTk.PhotoImage(logo)
        self.configurationTk.logo=logo
        canvas.create_image(self.configurationTk.winfo_screenwidth()/2, logo.height(), image=logo, anchor='center')


        #Buttons
        exitImg = Image.open('images/backButton.png')
        exitImg = exitImg.resize((int(self.configurationTk.winfo_screenheight()*.05), int(self.configurationTk.winfo_screenheight()*.05)), Image.ANTIALIAS)
        exitImg = ImageTk.PhotoImage(exitImg)
        self.configurationTk.exitImg=exitImg

        exitCanvas=canvas.create_image(self.configurationTk.winfo_screenwidth()*.05, self.configurationTk.winfo_screenheight()*.05, image=exitImg)
        canvas.tag_bind(exitCanvas, "<Button-1>",  (lambda _:closeTk()))

        #Labels Tk
        # labelimagen = Label(self.configurationTk, image=imagen)
        # labelimagen.pack()

        ##Detlimit
        detlimitFrame=LabelFrame(self.configurationTk, text='Detecciones de Espera EPP',bg='white')
        detlimitFrame.place(relx=.3, rely=.2, relwidth=.4, relheight=.1, )

        detlimitButton = Button(detlimitFrame, text="Cambiar limite de capturas", command=lambda:changeDetLimit(),bg='#CCEEFF')
        detlimitButton.place(relx=.5, rely=.8, anchor='center')

        self.detlimitLabel=Label(detlimitFrame, text='Value: {}'.format(self.detlimit), bg='white')
        self.detlimitLabel.place(relx=.5, rely=.4, anchor='center')

        ##Camera

        cameraFrame=LabelFrame(self.configurationTk, text='Seleccionar Camara',bg='white')
        cameraFrame.place(relx=.3, rely=.35, relwidth=.4, relheight=.1)

        cameraPPEButton = Button(cameraFrame, text="Configurar Camara PPE", command=lambda:self.configCameraPPETk(),bg='#CCEEFF')
        cameraPPEButton.place(relx=.4, rely=.8, anchor='center')

        cameraActionButton = Button(cameraFrame, text="Configurar Camara Planta", command=lambda:self.configCameraActionTk(),bg='#CCEEFF')
        cameraActionButton.place(relx=.6, rely=.8, anchor='center')

        self.cameraLabel=Label(cameraFrame, text='Camara PPE: {}                  Camara Planta: {}'.format(self.varCamera, self.actiondet.varCamera), bg='white')
        self.cameraLabel.place(relx=.5, rely=.4, anchor='center')

        ##PPE Image Test

        ppeimageFrame=LabelFrame(self.configurationTk, text='Test Imagen EPP',bg='white')
        ppeimageFrame.place(relx=.3, rely=.5, relwidth=.4, relheight=.1)

        ppeimageButton = Button(ppeimageFrame, text="Seleccionar Imagen para Test EPP", command=lambda:self.folderPpeframeSelect(),bg='#CCEEFF')
        ppeimageButton.place(relx=.5, rely=.8, anchor='center')

        self.ppeimageLabel=Label(ppeimageFrame, text='{}'.format(ppeimagetext()), bg='white')
        self.ppeimageLabel.place(relx=.5, rely=.4, anchor='center')

        ##PPE Video Test

        ppevideoFrame=LabelFrame(self.configurationTk, text='Test Video EPP',bg='white')
        ppevideoFrame.place(relx=.3, rely=.65, relwidth=.4, relheight=.1)

        ppevideoButton = Button(ppevideoFrame, text="Seleccionar Video para Test EPP", command=lambda:self.folderPpevideoSelect(),bg='#CCEEFF')
        ppevideoButton.place(relx=.5, rely=.8, anchor='center')

        self.ppevideoLabel=Label(ppevideoFrame, text='{}'.format(ppevideotext()), bg='white')
        self.ppevideoLabel.place(relx=.5, rely=.4, anchor='center')    

        ##Action Video Test

        actionvideoFrame=LabelFrame(self.configurationTk, text='Test Video Action',bg='white')
        actionvideoFrame.place(relx=.3, rely=.8, relwidth=.4, relheight=.1)

        actionvideoButton = Button(actionvideoFrame, text="Seleccionar Video para Test Action", command=lambda:self.folderactionvideoSelect(),bg='#CCEEFF')
        actionvideoButton.place(relx=.5, rely=.8, anchor='center')

        self.actionvideoLabel=Label(actionvideoFrame, text='{}'.format(actionvideotext()), bg='white')
        self.actionvideoLabel.place(relx=.5, rely=.4, anchor='center')  


        

        # closeWindow = Button(self.configurationTk, text="Cerrar Ventana", command=lambda:closeTk())
        # closeWindow.pack()

        # self.configurationTk.focus_force()
        # self.configurationTk.lift()
        # self.configurationTk.attributes('-topmost', True)
        # self.configurationTk.after_idle(self.configurationTk.attributes,'-topmost',False)

        self.configurationTk.lift()
        self.configurationTk.attributes('-topmost', True)
        self.configurationTk.after_idle(self.configurationTk.attributes,'-topmost',False)
        # adminConfigTk.withdraw()
        

    def adminConfigTk(self, user):
        adminConfigTk = Toplevel()
        adminConfigTk.title("Admin panel")
        adminConfigTk.resizable(False,False)
        adminConfigTk.config(background="#cceeff")
        # adminConfigTk.resizable(False,False)
        adminConfigTk.overrideredirect(True)
        adminConfigTk.geometry(f'{adminConfigTk.winfo_screenwidth()}x{adminConfigTk.winfo_screenheight()}')
        self.center_window(adminConfigTk)
        # adminConfigTk.geometry('300x300')
        # adminConfigTk.lift()
        # adminConfigTk.attributes('-topmost', True)

        def logout(user):
            API_Services.logout(user.getToken(), user.getRefreshToken())
            del user
            self.root.deiconify()
            self.root.focus_force()
            adminConfigTk.after(1000,adminConfigTk.destroy)


        #Canvas
        canvas = Canvas(adminConfigTk, borderwidth=0,highlightthickness=0)
        canvas.place(relx=.5, rely=.5, relwidth=1, relheight=1, anchor='center')

        bg = Image.open('images/network_bg.png')
        bg = bg.resize((adminConfigTk.winfo_screenwidth(), adminConfigTk.winfo_screenheight()), Image.ANTIALIAS)
        bg = ImageTk.PhotoImage(bg)
        adminConfigTk.bg=bg
        canvas.create_image(adminConfigTk.winfo_screenwidth()/2, adminConfigTk.winfo_screenheight()/2, image=bg)

        logo = Image.open('images/logo_hidrolatina.png')
        logo = logo.resize((325, 97), Image.ANTIALIAS)
        logo = ImageTk.PhotoImage(logo)
        adminConfigTk.logo=logo
        canvas.create_image(adminConfigTk.winfo_screenwidth()/2, logo.height(), image=logo, anchor='center')

        adminImg = Image.open('images/adminPanel.png')
        adminImg = adminImg.resize((325, 97), Image.ANTIALIAS)
        adminImg = ImageTk.PhotoImage(adminImg)
        adminConfigTk.adminImg=adminImg
        canvas.create_image(0, logo.height(), image=adminImg, anchor='w')

        sessionImg = Image.open('images/Session.png')
        sessionImg = sessionImg.resize((325, 97), Image.ANTIALIAS)
        sessionImg = ImageTk.PhotoImage(sessionImg)
        adminConfigTk.sessionImg=sessionImg
        canvas.create_image(adminConfigTk.winfo_screenwidth(), sessionImg.height(), image=sessionImg, anchor='e')

        canvas.create_text(adminConfigTk.winfo_screenwidth()-sessionImg.width()*.85,sessionImg.height(),anchor='w',fill='white', font="Digital-7 16 italic bold",
                        text="Usuario:   {}\nNombre:   {} {}".format(user.getUsername(), user.getName(), user.getLast_name()))

        #Labels
        
        usersImg = Image.open('images/usersButton2.png')
        usersImg = usersImg.resize((int(adminConfigTk.winfo_screenheight()*.2), int(adminConfigTk.winfo_screenheight()*.2)), Image.ANTIALIAS)
        usersImg = ImageTk.PhotoImage(usersImg)
        adminConfigTk.usersImg=usersImg

        usersCanvas=canvas.create_image(adminConfigTk.winfo_screenwidth()*.3, adminConfigTk.winfo_screenheight()*.3, image=usersImg)
        canvas.tag_bind(usersCanvas, "<Button-1>",  (lambda _:self.userManagementTk(user)))

         # Labels and Buttons


        configImg = Image.open('images/configButton2.png')
        configImg = configImg.resize((int(adminConfigTk.winfo_screenheight()*.2), int(adminConfigTk.winfo_screenheight()*.2)), Image.ANTIALIAS)
        configImg = ImageTk.PhotoImage(configImg)
        adminConfigTk.configImg=configImg

        configCanvas=canvas.create_image(adminConfigTk.winfo_screenwidth()*.5, adminConfigTk.winfo_screenheight()*.3, image=configImg)
        canvas.tag_bind(configCanvas, "<Button-1>",  (lambda _:self.openConfigurationTk(user, adminConfigTk)))

        exitImg = Image.open('images/exitButton2.png')
        exitImg = exitImg.resize((int(adminConfigTk.winfo_screenheight()*.2), int(adminConfigTk.winfo_screenheight()*.2)), Image.ANTIALIAS)
        exitImg = ImageTk.PhotoImage(exitImg)
        adminConfigTk.exitImg=exitImg 

        exitCanvas=canvas.create_image(adminConfigTk.winfo_screenwidth()*.7, adminConfigTk.winfo_screenheight()*.3, image=exitImg)
        canvas.tag_bind(exitCanvas, "<Button-1>",  (lambda _:logout(user)))

        
        eppImg = Image.open('images/eppdetButton2.png')
        eppImg = eppImg.resize((int(adminConfigTk.winfo_screenheight()*.2), int(adminConfigTk.winfo_screenheight()*.2)), Image.ANTIALIAS)
        eppImg = ImageTk.PhotoImage(eppImg)
        adminConfigTk.eppImg=eppImg

        eppCanvas=canvas.create_image(adminConfigTk.winfo_screenwidth()*.4, adminConfigTk.winfo_screenheight()*.6, image=eppImg)
        canvas.tag_bind(eppCanvas, "<Button-1>",  (lambda _:self.showPytorchCameraTk(user)))

        actImg = Image.open('images/actdetButton2.png')
        actImg = actImg.resize((int(adminConfigTk.winfo_screenheight()*.2), int(adminConfigTk.winfo_screenheight()*.2)), Image.ANTIALIAS)
        actImg = ImageTk.PhotoImage(actImg)
        adminConfigTk.actImg=actImg

        eppCanvas=canvas.create_image(adminConfigTk.winfo_screenwidth()*.6, adminConfigTk.winfo_screenheight()*.6, image=actImg)
        canvas.tag_bind(eppCanvas, "<Button-1>",  (lambda _:self.showActionsTk()))
  
        
        adminConfigTk.focus_force()
        adminConfigTk.lift()
        adminConfigTk.attributes('-topmost', True)
        adminConfigTk.after(100,self.root.withdraw)
        adminConfigTk.after_idle(adminConfigTk.attributes,'-topmost',False)
      