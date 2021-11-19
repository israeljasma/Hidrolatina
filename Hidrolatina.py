from datetime import datetime, timedelta
from sys import path
from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox, filedialog, simpledialog, Listbox
import cv2
import os
import platform
import time
from threading import Thread, Lock
from effdet.utils.inference import init_effdet_model,inference_effdet_model

from Services import API_Services
from UserClass import Person
from FileManagementClass import FileManagement
from NFCClass import NFC

from numpy import CLIP
from imagenClipClass import imageClip

import torch

##Path
if platform.system() == "Darwin":
    print("MacOS")
    import getpass
    username = getpass.getuser()
    DATA_DIR = os.path.join("/Users/" + username + "/Documents/hidrolatina")
    MODELS_DIR = os.path.join(DATA_DIR, "models")
    dir = [DATA_DIR, MODELS_DIR]
    for dir in [DATA_DIR, MODELS_DIR]:
        if not os.path.exists(dir):
            os.makedirs(dir)
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

def downloadEfficientDet():
    url = 'https://github.com/EquipoVandV/VandVEfficientDet/archive/refs/heads/main.zip'
    FileManagement.extractFile(FileManagement.downloadFile(url, DATA_DIR), DATA_DIR)
    return

def importMDETER():
    import requests
    import torchvision.transforms as T
    from collections import defaultdict
    import pathlib
    torch.set_grad_enabled(False);
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=True, return_postprocessor=True)
    model = model.cuda()
    model.eval();

    global transform, box_cxcywh_to_xyxy, rescale_bboxes, COLORS, apply_mask, plot_results, id2answerbytype, plot_inference, plot_inference_qa
    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    import json
    answer2id_by_type = json.load(requests.get("https://nyu.box.com/shared/static/j4rnpo8ixn6v0iznno2pim6ffj3jyaj8.json", stream=True).raw)
    id2answerbytype = {}                                                       
    for ans_type in answer2id_by_type.keys():                        
        curr_reversed_dict = {v: k for k, v in answer2id_by_type[ans_type].items()}
        id2answerbytype[ans_type] = curr_reversed_dict 


    def plot_inference(im, caption):
    # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0).cuda()

        # propagate through the model
        memory_cache = model(img, [caption], encode_and_save=True)
        outputs = model(img, [caption], encode_and_save=False, memory_cache=memory_cache)

        global probas, keep
        # keep only predictions with 0.7+ confidence
        probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
        keep = (probas > 0.7).cpu()

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)

        # Extract the text spans predicted by each box
        positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
        predicted_spans = defaultdict(str)
        for tok in positive_tokens:
            item, pos = tok
            if pos < 255:
                span = memory_cache["tokenized"].token_to_chars(0, pos)
                predicted_spans [item] += " " + caption[span.start:span.end]
        labels = [predicted_spans [k] for k in sorted(list(predicted_spans .keys()))]
        global bboxes
        bboxes=bboxes_scaled.numpy()
        # print('boxes: ', bboxes)
        # plot_results(im, probas[keep], bboxes_scaled, labels)

    print("MDETR cargado")

def clearCacheMDETR():
    torch.cuda.empty_cache()

def loadEfficient():
    weigths_effdet = 'C:/hidrolatina/EfficientDetVandV-main/effdet/logs/person_coco/efficientdet-d2_58_8260_best.pth'
    global obj_list
    obj_list = ['person']
    global model_effdet
    model_effdet = init_effdet_model(weigths_effdet, obj_list)
    ##Class CameraStream
    global CameraStream
    class CameraStream(object):
        def __init__(self, src=0):
            self.stream = cv2.VideoCapture(src)

            (self.grabbed, self.frame) = self.stream.read()
            self.started = False
            self.read_lock = Lock()

        def start(self):
            if self.started:
                print("already started!!")
                return None
            self.started = True
            self.thread = Thread(target=self.update, args=())
            self.thread.start()
            return self

        def update(self):
            while self.started:
                (grabbed, frame) = self.stream.read()
                self.read_lock.acquire()
                self.grabbed, self.frame = grabbed, frame
                self.read_lock.release()

        def read(self):
            self.read_lock.acquire()
            frame = self.frame.copy()
            self.read_lock.release()
            return frame

        def stop(self):
            self.started = False
            self.stream.release()
        

        def __exit__(self, exc_type, exc_value, traceback):
            self.thread.join()

    print('EfficientDET Cargado')

def MDETR(im):
    plot_inference(im, "a hand")
    im_hand=im.crop((bboxes[argmax(probas[keep])]))

    plot_inference(im, "a head")
    im_head=im.crop((bboxes[argmax(probas[keep])]))

    plot_inference(im, "a boot")
    im_boot=im.crop((bboxes[argmax(probas[keep])]))

    objectListMDETR= {'im_head':im_head, 'im_hand': im_hand, 'im_boot': im_boot}
    return objectListMDETR

def loadClip():
    global clipit
    import clip as clipit
    import glob

    global candidate_captions
    candidate_captions={'im_head': [['a head with a yellow helmet','Just a head'], ['Head with headphones', 'Just a head'],['a Head with goggles', 'Just a head'],['Head with a medical mask', 'Just a head']],
                'im_hand':[['A blue hand', 'A pink hand']],
                'im_boot':[['A black boot', 'A shoe']]}
    # candidate_captions={'im_head': [['a white hat','A head'], ['a big headset', 'a face'],['a face with glasses', 'A head'],['Mask', 'Just a head']],
    #             'im_hand':[['A blue hand', 'A pink hand']],
    #             'im_boot':[['a large boot', 'a small shoe']]}
    global names_ppe
    names_ppe = {'im_head': ['Casco', 'Audífonos', 'Antiparras', 'Mascarilla'], 'im_hand': ['Guantes'], 'im_boot': ['Botas']}

    global argmax
    def argmax(iterable):
        return max(enumerate(iterable), key=lambda x: x[1])[0]

    ##################Arreglar global##################
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    ##################Arreglar global##################
    global modelc, process
    modelc, process = clipit.load("ViT-B/32", device=device)

    ##################Arreglar global##################
    global nstr
    def nstr(obj):
        return [name for name in globals() if globals()[name] is obj][0]
    print('Clip Cargado')

def clip(bodypart, mdetr_list):
    pred_clip=[]
    for i in range(len(candidate_captions[bodypart])):
        text = clipit.tokenize(candidate_captions[bodypart][i]).to(device)
        image = process(mdetr_list[bodypart]).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = modelc.encode_image(image)
            text_features = modelc.encode_text(text)
            
            logits_per_image, logits_per_text = modelc(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            # pred = class_names[argmax(list(probs)[0])]
            if argmax(list(probs)[0])== 0:
                pred_clip.append('OK')
            else:
                pred_clip.append('NO DETECTADO')

    return pred_clip

# def librerias():
    # importMDETER()
    # loadClip()
    # loadEfficient()
    # messagebox.showinfo(message="Dependencias cargadas")
    # thread.join()


def loadALL():
    # from threading import Thread
    # libThread= Thread(target=librerias, args=(),daemon=True)
    # libThread.start()
    importMDETER()
    loadClip()
    loadEfficient()
    messagebox.showinfo(message="Dependencias cargadas")


########Windows#######

#Def
def popup(message):
    messagebox.showinfo(message=message)

def folderSelect():
    folder_selected = filedialog.askdirectory()
    print(folder_selected)

def folderframeSelect():
    global frame_selected
    frame_selected = filedialog.askopenfilename()
    print(frame_selected)

###################Def Windows's###################

def showPytorchCameraTk(user):
    # import datetime
    import numpy as np

    #Var/Global
    global det
    global image
    global original_image
    obj_list = ['person']
    det=0

    #Tkinter config
    pytorchCameraTk = Toplevel()
    pytorchCameraTk.title('Camara')
    # pytorchCameraTk.resizable(False,False)
    pytorchCameraTk.config(background="#cceeff")
    pytorchCameraTk.overrideredirect(True)
    pytorchCameraTk.geometry(f'{pytorchCameraTk.winfo_screenwidth()}x{pytorchCameraTk.winfo_screenheight()}')
    # pytorchCameraTk.geometry(f'{1280}x{720}')
 
    # pytorchCameraTk.geometry("1280x720")

    image = PhotoImage(file="white-image.png")
    original_image = image.subsample(1,1)

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
    Label(imageHeadFrame, image=original_image).grid(row=1, column=0, padx=5, pady=5)

    # ###Label imageHandFrame Sub frame lvl 2 handFrame
    Label(imageHandFrame, image=original_image).grid(row=1, column=0, padx=5, pady=5)

    # ###Label imagebootFrame Sub frame lvl 2 bootFrame
    Label(imageBootFrame, image=original_image).grid(row=1, column=0, padx=5, pady=5)

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
    def closeTk():
        #Destroy window
        cap.stop()
        pytorchCameraTk.destroy()
        # root.deiconify()

    def testFrame():
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(cv2.resize(frame, (round(camWidth), round(camHeight))), cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        labelVideo.imgtk = imgtk
        labelVideo.configure(image=imgtk)
        labelVideo.after(10, testFrame)

        global det
        det = det+1
        if det > 60:
            print('Rseset det to 0')
            # updateLabelTest()
            det = 0

    def showFrame():
        # _, frame = cap.read()
        # frame = cv2.flip(frame, 1)
        try:
            frame=cv2.imread(frame_selected)
        except:
            frame = cap.read()

        out = inference_effdet_model(model_effdet, frame)
        ori_img = frame.copy()

        for j in range(len(out['bbox'])):
            (x1, y1, x2, y2) = out['bbox'][j].astype(np.int)
            cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[out['class_ids'][j]]
            score = float(out['scores'][j])

            cv2.putText(ori_img, '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (255, 255, 0), 2)

        cv2image = cv2.cvtColor(cv2.resize(ori_img, (600, 500)), cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        labelVideo.imgtk = imgtk
        labelVideo.configure(image=imgtk)

        global det
        print(det)
        if len(out['class_ids']) == 0:
            det = 0
        if len(out['class_ids']) > 0:
            det += 1
            if det==20:

                print("Reset")
                for i in range((out['scores']).size):
                    detected_boxes= out['bbox'][i]

                # Crop and save detedtec bounding box image

                xmin = int((detected_boxes[0]))
                ymin = int((detected_boxes[1]))
                xmax = int((detected_boxes[2]))
                ymax = int((detected_boxes[3]))
                cropped_img =frame[ymin:ymax,xmin:xmax]


                global im
                im = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                print(im)
                cap.stop()
                copy_imgtk = imgtk
                labelVideo.imgtk = copy_imgtk
                mdetr_list=MDETR(im)
    ################################################ CORRERGIR ###############################################
    ################################################ CORRERGIR ###############################################
    ################################################ CORRERGIR ###############################################
                # print(mdetr_list)
                global listImagenClip
                listImagenClip = []
                for bodypart in mdetr_list.keys():
                    listImagenClip.append(imageClip(names_ppe[bodypart], ImageTk.PhotoImage(mdetr_list[bodypart].resize((150,150))), clip(bodypart, mdetr_list)))
                    # print(bodypart)
                
                updateLabel()
    ################################################ CORRERGIR ###############################################
    ################################################ CORRERGIR ###############################################
    ################################################ CORRERGIR ###############################################
                
        if det<20:
            labelVideo.after(10, showFrame)

    def counterPopUp(endTime, booleanAnswer):
        if datetime.now() > endTime:
            print('si')
            print(datetime.now().strftime('%H:%M:%S'), endTime.strftime('%H:%M:%S'))
            print('funciona')
            popupIdentificationTk(booleanAnswer)
        else:
            print('no')
            print(datetime.now().strftime('%H:%M:%S'), endTime.strftime('%H:%M:%S'))
            pytorchCameraTk.after(5000, counterPopUp, endTime, booleanAnswer)

    def updateLabel():
        global image
        global original_image

        image = PhotoImage(file="logo-sm.png")
        original_image = image.subsample(1,1)

        #Head Frame
        Label(imageHeadFrame, image=(listImagenClip[0].getImage())).grid(row=1, column=0, padx=5, pady=5)
        Label(dataHeadFrame, text=(listImagenClip[0].getAnswer()[0]), width=15).grid(row=0, column=1, padx=5, pady=5)
        Label(dataHeadFrame, text=(listImagenClip[0].getAnswer()[1]), width=15).grid(row=1, column=1, padx=5, pady=5)
        Label(dataHeadFrame, text=(listImagenClip[0].getAnswer()[2]), width=15).grid(row=2, column=1, padx=5, pady=5)
        Label(dataHeadFrame, text=(listImagenClip[0].getAnswer()[3]), width=15).grid(row=3, column=1, padx=5, pady=5)

        #Hand Frame
        Label(imageHandFrame, image=(listImagenClip[1].getImage())).grid(row=1, column=0, padx=5, pady=5)
        Label(dataHandFrame,  text=(listImagenClip[1].getAnswer()[0]), width=15).grid(row=0, column=1, padx=5, pady=5)

        #Boot Frame
        Label(imageBootFrame, image=(listImagenClip[2].getImage())).grid(row=1, column=0, padx=5, pady=5)
        Label(dataBootFrame, text=(listImagenClip[2].getAnswer()[0]), width=15).grid(row=0, column=1, padx=5, pady=5)

        booleanAnswer = None
        for list in listImagenClip:
            for j in range(len(list.getAnswer())):
                if list.getAnswer()[j] == 'OK':
                    print(list.getName()[j])
                    print(list.getAnswer()[j])
                    booleanAnswer = True
                else:
                    booleanAnswer = False
                print(booleanAnswer)

        endTime = datetime.now() + timedelta(seconds=15)
        if len(listImagenClip) > 0:
            counterPopUp(endTime, booleanAnswer)
        # pytorchCameraTk.after(1, counterPopUp, endTime, booleanAnswer)

    # testFrame()
    exitButton = Button(pytorchCameraTk, text='Cerrar ventana', command=closeTk)
    exitButton.grid(row=1, column=0)

    testButtonUpdate = Button(pytorchCameraTk, text='Test Update', command=updateLabel)
    testButtonUpdate.grid(row=1, column=1)
    showFrame()


def configCameraTk(configurationTk):
    # Config tk
    configCameraTk = Toplevel()
    configCameraTk.resizable(False,False)
    configCameraTk.protocol("WM_DELETE_WINDOW", exit)
    configCameraTk.title("Configuracion camaras")
    # configCameraTk.overrideredirect(True)

    width = 200
    height = 300
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

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

def nfc_identifyTk():
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
        root.deiconify()
    
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
    imageWaitDetectionLeft = Image.open('images/waiting_identification_left.png')
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

def popupIdentificationTk(booleanAnswer):
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

def userManagementTk(user):
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

def openConfigurationTk(user, adminConfigTk):
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
    buttonDirectory = Button(configurationTk, text="Cambiar directorio", command=folderSelect)
    buttonDirectory.pack()

    buttonThreshold = Button(configurationTk, text="Cambiar Threshold", command=lambda:changeThreshold())
    buttonThreshold.pack()

    buttonIou_threshold = Button(configurationTk, text="Cambiar Iou Threshold", command=lambda:changeIouThreshold())
    buttonIou_threshold.pack()

    buttonDetLimit = Button(configurationTk, text="Cambiar limite de capturas", command=lambda:changeDetLimit())
    buttonDetLimit.pack()

    buttonClass = Button(configurationTk, text="Cambiar clases")
    buttonClass.pack()

    buttonClass = Button(configurationTk, text="Configurar Camaras", command=lambda:configCameraTk(configurationTk))
    buttonClass.pack()

    buttonfDirectory = Button(configurationTk, text="ImagenTest", command=lambda:folderframeSelect())
    buttonfDirectory.pack()

    closeWindow = Button(configurationTk, text="Cerrar Ventana", command=lambda:closeTk())
    closeWindow.pack()

def adminConfigTk(user):
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

    testButton = Button(adminConfigTk, text='Test Camara',command=showPytorchCameraTk, fg='red').grid()
    testButton = Button(adminConfigTk, text='Test download',command=downloadEfficientDet, fg='red').grid()
    testButton = Button(adminConfigTk, text='Test NFC',command=nfc_identifyTk, fg='red').grid()
    testButton = Button(adminConfigTk, text='Test POPUP',command=popupIdentificationTk, fg='red').grid()
    testButton = Button(adminConfigTk, text='Cargar Dependencias',command=loadALL, fg='red').grid()

    createUser = Button(adminConfigTk, text='Gestion de usuario', command=lambda:userManagementTk(user))
    createUser.grid()

    configButton = Button(adminConfigTk, command=lambda:openConfigurationTk(user, adminConfigTk), text='Configuraciones')
    configButton.grid()

    exitButton = Button(adminConfigTk, text="Cerrar Sesion", command=lambda:logout(user))
    exitButton.grid()


############ Start App ############
root = Tk()
root.geometry('350x500+500+50')
root.resizable(0,0)
root.config(bg='#CCEEFF')
root.title('Hidrolatina')

# Def
def verification():
    user = userEntry.get()
    password = passwordEntry.get()
    try:
        person = API_Services.login(user, password)
        if 'token' in person:
            user = Person(person['user']['username'], person['user']['name'], person['user']['last_name'], person['user']['email'], person['token'])
            #Hide Root Window
            # root.withdraw()
            adminConfigTk(user)
        else:
            messagebox.showinfo(message=person['error'], title="Login")
    except:
        messagebox.showerror(title='Error de conexión', message='No se ha podido establecer una conexión con el servidor. Comuníquese con su encargado de TI.')
    

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