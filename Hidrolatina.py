from sys import path
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
from tkinter import filedialog
import sqlite3
import os
import platform

##Download EfficientDET import's
import tarfile
import urllib.request

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

def efficientDETModels(MODELS_DIR, selected):
    MODEL_DATE = '20200711'
    MODEL_NAME = 'efficientdet_'+selected+'_coco17_tpu-32'
    MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
    MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
    MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
    PATH_TO_MODEL_TAR = os.path.join(MODELS_DIR, MODEL_TAR_FILENAME)
    ##################Arreglar PATH_TO_CKPT##################
    global PATH_TO_CKPT
    PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
    ##################Arreglar PATH_TO_CFG##################
    global PATH_TO_CFG
    PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
    if not os.path.exists(PATH_TO_CKPT):
        print('Downloading model. This may take a while... ', end='')
        urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
        tar_file = tarfile.open(PATH_TO_MODEL_TAR)
        tar_file.extractall(MODELS_DIR)
        tar_file.close()
        os.remove(PATH_TO_MODEL_TAR)
        print('Done')
    
    # Download labels file
    LABEL_FILENAME = 'mscoco_label_map.pbtxt'
    LABELS_DOWNLOAD_BASE = \
        'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
    if not os.path.exists(PATH_TO_LABELS):
        print('Downloading label file... ', end='')
        urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
        print('Done')

def importMDETER():
    from PIL import Image
    import requests
    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import torch.nn.functional as F
    import numpy as np
    from skimage.measure import find_contours
    from matplotlib import patches,  lines
    from matplotlib.patches import Polygon
    import pathlib
    torch.set_grad_enabled(False);
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    model_qa = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5_gqa', pretrained=True, return_postprocessor=False)
    model_qa = model_qa.cuda()
    model_qa.eval();
    model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=True, return_postprocessor=True)
    model = model.cuda()
    model.eval();

def clearCacheMDETR():
    torch.cuda.empty_cache()

def loadODAPI():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
    import tensorflow as tf
    from object_detection.utils import label_map_util
    from object_detection.utils import config_util
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.builders import model_builder

    tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

    # # Enable GPU dynamic memory allocation
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)

    #Set CPU
    tf.config.set_visible_devices([], 'GPU')

    # Load pipeline config and build a detection model
    print(PATH_TO_CFG)
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])


def loadEfficient():
    ### Arreglar directorio
    import sys
    sys.path.append("C:/Users/Doravan/Desktop/Hidrolatina/torchtest/Yet-Another-EfficientDet-Pytorch")
    # os.chdir('C:/Users/Doravan/Desktop/Hidrolatina/torchtest/Yet-Another-EfficientDet-Pytorch')
    from torch.backends import cudnn
    from backbone import EfficientDetBackbone
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from efficientdet.utils import BBoxTransform, ClipBoxes
    from utils.utils import preprocess, invert_affine, postprocess

    ##################Arreglar global##################
    global preprocess, invert_affine, postprocess, BBoxTransform, ClipBoxes


    compound_coef = 2
    force_input_size = None  # set None to use default size

    ##################Arreglar global##################
    global use_cuda, use_float16
    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    obj_list = ['person']

    ##################Arreglar global##################
    global input_size
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    ##################Arreglar global##################
    global model
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),

                                # replace this part with your project's anchor config
                                ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                                scales=[2 * 0, 2 * (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    # model.load_state_dict(torch.load('logs/person - copia/efficientdet-d1_95_2200.pth'))
    model.load_state_dict(torch.load('C:/Users/Doravan/Desktop/Hidrolatina/torchtest/Yet-Another-EfficientDet-Pytorch/efficientdet-d2_58_8260_best.pth'))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

def pytorchCamera():
    import cv2
    import matplotlib.pyplot as plt
    import datetime
    det=0

    cap = cv2.VideoCapture(0)
    import numpy as np

    obj_list = ['person']

    while True:
        # Read frame from camera
        cap.set(cv2.CAP_PROP_FPS,16)
        ret, image_np = cap.read()
        image_path=[image_np]

        
        threshold = 0.5
        iou_threshold = 0.5

        # # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # image_np_expanded = np.expand_dims(image_np, axis=0)
        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_size)

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, iou_threshold)

        out = invert_affine(framed_metas, out)


        # if len(out[0]['rois']) == 0:

        ori_img = ori_imgs[0].copy()
        for j in range(len(out[0]['rois'])):
            (x1, y1, x2, y2) = out[0]['rois'][j].astype(np.int)
            cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[out[0]['class_ids'][j]]
            score = float(out[0]['scores'][j])

            cv2.putText(ori_img, '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (255, 255, 0), 2)

        cv2.imshow('object_detection', cv2.resize(ori_img, (800, 600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        if len(out[0]['scores']) > 0:
            det += 1
            if det==5:     #break in det-1

                now = datetime.datetime.now()
                date_hour='%d/%d/%d-%d:%d:%d'%( now.day, now.month, now.year, now.hour, now.minute, now.second )
                print(date_hour)
                cap.release()
                cv2.destroyAllWindows()
                break

        if len(out[0]['scores'])==0:
            det=0


        
        # Print objects detected's labels
        # score_index= np.where(score > score_thresh)[0]
        # class_index= classes[score_index]

        for i in range((out[0]['scores']).size):
            detected_boxes= out[0]['rois'][i]
            # detected_labels= category_index[class_index[i-1]]['name']
                


        # Crop and save detedtec bounding box image
            # (frame_height, frame_width) = ori_img.shape[:2]
            # ymin = int((detected_boxes[0]*frame_height))
            # xmin = int((detected_boxes[1]*frame_width))
            # ymax = int((detected_boxes[2]*frame_height))
            # xmax = int((detected_boxes[3]*frame_width))
            xmin = int((detected_boxes[0]))
            ymin = int((detected_boxes[1]))
            xmax = int((detected_boxes[2]))
            ymax = int((detected_boxes[3]))
            cropped_img = image_np[ymin:ymax,xmin:xmax]
            global imagencamera
            if not cropped_img.size == 0:
                imagencamera = cropped_img
            # imgplot = plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            # plt.show()
            # print(imgplot)
            # print('ymin',ymin)
            # print('fdsfssdfdsf')
            # print('detected_boxes',detected_boxes)
            # print('cropped_img',cropped_img)
            if cropped_img.size == 0:
                continue
            else:
                #cv2.imwrite('cropped_image_{}.jpg'.format(i), cropped_img)
                print(cropped_img)
            # imagencamera = Image.fromarray(cropped_img, 'RGB')
            

def prefunctionclip():
    class_names=['mask and headphones', 'mask and goggles ','mask, headphones and goggles','mask', 'goggles', 'no_mask']
    class_names
    im = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plot_inference(im, "a hand")
    im_hand=im.crop((bboxes[len(bboxes)-1]))
    plot_inference(im, "a tiny head")
    im_head=im.crop((bboxes[len(bboxes)-1]))
    im_head.save('clip/cropped_head.jpg')
    plot_inference_qa(im_hand, "what color are the fingers?")

def clip():
    import clip
    import glob

    def argmax(iterable):
        return max(enumerate(iterable), key=lambda x: x[1])[0]

    ##################Arreglar global##################
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    ##################Arreglar global##################
    global modelc, transform
    modelc, transform = clip.load("ViT-B/32", device=device)

    ##################Arreglar global##################
    global text
    text = clip.tokenize(candidate_captions).to(device)

def algunafuncion():
    head=Image.open('cropped_head.jpg')
    image = transform(head).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = modelc.encode_image(image)
        text_features = modelc.encode_text(text)
        
        logits_per_image, logits_per_text = modelc(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        pred = class_names[argmax(list(probs)[0])]
        print(pred)
        np_image = np.array(head)
    plt.imshow(np_image)

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

def popup(message):
    messagebox.showinfo(message=message)

def folderSelect():
    folder_selected = filedialog.askdirectory()
    print(folder_selected)

###################Def Windows's###################

def openDownloadModelsTk():
    #Configurations of windows
    downloadModelsTk = Toplevel()
    downloadModelsTk.resizable(False,False)
    downloadModelsTk.protocol("WM_DELETE_WINDOW", exit)
    downloadModelsTk.title("Descargar Modelos")

    #Dimensions of windows downloadModelsTk
    width = 200
    height = 200
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width/2) - (app_width/2)
    y = (screen_height/2) - (app_height/2)

    downloadModelsTk.geometry(f'{width}x{height}+{int(x)}+{int(y)}')

    #Def
    def closeTk():
        downloadModelsTk.destroy()
        root.deiconify()
    
    def switchSelection(selectedOption):
        
        if selectedOption == 'EfficientDET D1':
            efficient = 'd1'
            return efficient
        elif selectedOption == 'EfficientDET D2':
            efficient = 'd2'
            print('d2')
            return efficient
        elif selectedOption == 'EfficientDET D3':
            efficient = 'd3'
            print('d3')
            return efficient
        elif selectedOption == 'EfficientDET D4':
            efficient = 'd4'
            print('d4')
            return efficient
        elif selectedOption == 'EfficientDET D5':
            efficient = 'd5'
            print('d5')
            return efficient
        elif selectedOption == 'EfficientDET D6':
            efficient = 'd6'
            print('d6')
            return efficient
        elif selectedOption == 'EfficientDET D7':
            efficient = 'd7'
            print('d7')
            return efficient
        elif selectedOption == 'Todos':
            efficient = 'Todos'
            print('No implementado')
            message = 'No implementado, elija otra opción'
            popup(message)
            # return efficient
        else:
            print('Seleccione opcion')
            message = 'Seleccione una opcion valida'
            popup(message)

    def download():
        #efficientDETModels
        print(clicked.get())
        efficientDETModels(MODELS_DIR, switchSelection(clicked.get()))
    
    
    #Hide Root Window
    root.withdraw()

    #Dropdown Menu
    selectLabel = Label(downloadModelsTk, text="EfficientDET a descargar")
    selectLabel.pack()

    option = [
        'Selecione una opcion',
        'EfficientDET D1',
        'EfficientDET D2',
        'EfficientDET D3',
        'EfficientDET D4',
        'EfficientDET D5',
        'EfficientDET D6',
        'EfficientDET D7',
        'Todos'
    ]
    option3 = ['yeeey']

    option2 = {'yeeey': 'd0',
        'EfficientDET D1': 'd1',
        'EfficientDET D2' : 'd2',
        'EfficientDET D3' : 'd3',
        'EfficientDET D4' : 'd4',
        'EfficientDET D5' : 'd5',
        'EfficientDET D6' : 'd6',
        'EfficientDET D7' : 'd7',
        'EfficientDET D8' : 'd8'}
    
    op = list(option2.keys())

    clicked = StringVar()
    clicked.set(option[0])
    drop = OptionMenu(downloadModelsTk, clicked, *option )
    drop.pack()

    pri = Button(downloadModelsTk, text='Descargar modelos', command=download)
    pri.pack()

    #Buttons

    closeWindow = Button(downloadModelsTk, text="Cerrar Ventana", command=closeTk)
    closeWindow.pack()

def clipImageTk():
    global imagenn
    clipImageTkinter = Toplevel()
    # clipImageTkinter.resizable(False,False)
    clipImageTkinter.title("Imagen Clip")
    # im = Image.fromarray(imagencamera)


    # width = im.width()
    # height = (im.height() + 100)
    # screen_width = root.winfo_screenwidth()
    # screen_height = root.winfo_screenheight()

    # x = (screen_width/2) - (app_width/2)
    # y = (screen_height/2) - (app_height/2)

    # clipImageTkinter.geometry(f'{width}x{height}+{int(x)}+{int(y)}')

    path1 = Image.open('c:/Users/Doravan/Desktop/unnamed.jpg')
    path2 = Image.open('c:/Users/Doravan/Desktop/pngwingcom.png')
    imagenn = ImageTk.PhotoImage(path1)
    iyyey = ImageTk.PhotoImage(path1)

    imagenList=[imagenn]
    labelimage = Label(clipImageTkinter, image=imagenList)
    labelimage.pack()

    # buttonBack = Button(clipImageTkinter, text='<<', command=lambda:back).pack()
    # buttonForward = Button(clipImageTkinter, text='>>', command=lambda:forward).pack()
    exitButton = Button(clipImageTkinter, text="Salir", command=lambda:clipImageTkinter.destroy())
    exitButton.pack()

def showImageClipTk():
    #Config Tk
    imageClipTk = Toplevel()
    imageClipTk.title('Imagenes')
    imageClipTk.resizable(False,False)
    imageClipTk.protocol("WM_DELETE_WINDOW", exit)
    imageClipTk.overrideredirect(True)
    x = root.winfo_x()
    y = root.winfo_y()
    imageClipTk.geometry("+%d+%d" % (x, y))

    #Global Declarations
    global imagenlist0
    global imagenList
    global labelimage
    global buttonBack
    global buttonForward

    #Def into tk

    def closeTk():
        imageClipTk.destroy()
        root.deiconify()

    def forward(imageNumber):
        #Global Declarations into Def tk
        global labelimage
        global buttonBack
        global buttonForward

        labelimage.grid_forget()
        labelimage = Label(imageClipTk, image=imagenList[imageNumber])
        buttonForward = Button(imageClipTk, text='>>', command=lambda:forward(imageNumber+1))
        buttonBack = Button(imageClipTk, text='<<', command=lambda:back(imageNumber-1))

        print(imageNumber)
        if imageNumber == len(imagenList)-1:
            buttonForward = Button(imageClipTk, text='>>', state=DISABLED)
        
        labelimage.grid(row=0, column=0, columnspan=3)
        buttonBack.grid(row=1, column=0)
        buttonForward.grid(row=1, column=2)

        return

    def back(imageNumber):
        global labelimage
        global buttonBack
        global buttonForward

        labelimage.grid_forget()
        labelimage = Label(imageClipTk, image=imagenList[imageNumber])
        buttonForward = Button(imageClipTk, text='>>', command=lambda:forward(imageNumber+1))
        buttonBack = Button(imageClipTk, text='<<', command=lambda:back(imageNumber-1))

        print(imageNumber)
        if imageNumber == 0:
            buttonBack = Button(imageClipTk, text='<<', state=DISABLED)
        
        labelimage.grid(row=0, column=0, columnspan=3)
        buttonBack.grid(row=1, column=0)
        buttonForward.grid(row=1, column=2)
        return

    #Hide Root Window
    root.withdraw()

    #Path Images
    imgpath1 = Image.open('c:/Users/Doravan/Desktop/unnamed.jpg')
    imgpath2 = Image.open('c:/Users/Doravan/Desktop/600x400.jpg')
    imgpath3 = Image.open('c:/Users/Doravan/Desktop/descarga.jpg')
    imgpath4 = Image.open('c:/Users/Doravan/Desktop/lzN5Fa.jpg')
    imgpath5 = Image.open('e:/Softmaking/Proyectos/Hidrolatina/valorant.jpg')
    imagenn = ImageTk.PhotoImage(imgpath2)

    #Images
    imagenlist0 = ImageTk.PhotoImage(imgpath1)
    imagenlist1 = ImageTk.PhotoImage(imgpath2)
    imagenlist2 = ImageTk.PhotoImage(imgpath3)
    imagenlist3 = ImageTk.PhotoImage(imgpath4)
    imagenlist4 = ImageTk.PhotoImage(imgpath5)

    #List Images
    imagenList = [imagenlist0, imagenlist1, imagenlist2, imagenlist3, imagenlist4]

    #Test imagenList
    # print(len(imagenList))

    #Label Tk
    labelimage = Label(imageClipTk, image=imagenList[0])
    labelimage.grid(row=0, column=0, columnspan=3)

    #Buttons Tk
    buttonBack = Button(imageClipTk, text='<<', command=lambda:back, state=DISABLED)
    buttonBack.grid(row=1, column=0)

    exitButton = Button(imageClipTk, text="Cerrar ventana", command=closeTk)
    exitButton.grid(row=1, column=1)

    buttonForward = Button(imageClipTk, text='>>', command=lambda:forward(1))
    buttonForward.grid(row=1, column=2)


def openConfigurationTk():
    global imagen
    # Config tk
    configurationTk = Toplevel()
    configurationTk.resizable(False,False)
    configurationTk.protocol("WM_DELETE_WINDOW", exit)
    configurationTk.title("Configuraciones")
    configurationTk.overrideredirect(True)

    imagen = ImageTk.PhotoImage(Image.open("E:/Softmaking/Proyectos/Hidrolatina/valorant.jpg"))

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

    closeWindow = Button(configurationTk, text="Cerrar Ventana", command=lambda:configurationTk.destroy())
    closeWindow.pack()

#Buttons
buttonFlase = Button(root, text="yeeey", command=printLabel).pack()
configButton = Button(root, text="Configuraciones", command=openConfigurationTk, fg="blue").pack()
downloadModels = Button(root, text="Modelos", command=openDownloadModelsTk).pack()

messagebuton = Button(root, text="Popup", command=popup).pack()

importLibraryButton = Button(root, text='Cargar librerias', command=importMDETER).pack()
clearMDETRyButton = Button(root, text='Limpiar MDETR', command=clearCacheMDETR).pack()
loadODAPIButton = Button(root, text='Cargar OD API', command=loadODAPI).pack()
loadEfficientIButton = Button(root, text='Efficient Pytorch', command=loadEfficient).pack()
pytorchCameraButton = Button(root, text='Pytorch Camara', command=pytorchCamera).pack()
imagenClipButton = Button(root, text='Imagen Clip', command=clipImageTk).pack()
showImageClipButton = Button(root, text='Imagen Clip', command=showImageClipTk).pack()


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