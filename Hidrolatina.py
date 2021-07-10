from glob import glob
from sys import path
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
from tkinter import filedialog
import sqlite3
import os
import platform
from imagenClipClass import imageClip

##Download EfficientDET import's
import tarfile
import urllib.request

import torch

listImagenClip = []

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

    def apply_mask(image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                    image[:, :, c] *
                                    (1 - alpha) + alpha * color[c] * 255,
                                    image[:, :, c])
        return image

    def plot_results(pil_img, scores, boxes, labels, masks=None):
        plt.figure(figsize=(16,10))
        np_image = np.array(pil_img)
        ax = plt.gca()
        colors = COLORS * 100
        if masks is None:
            masks = [None for _ in range(len(scores))]
        assert len(scores) == len(boxes) == len(labels) == len(masks)
        for s, (xmin, ymin, xmax, ymax), l, mask, c in zip(scores, boxes.tolist(), labels, masks, colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
            text = f'{l}: {s:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

            if mask is None:
                continue
            np_image = apply_mask(np_image, mask, c)

            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=c)
                ax.add_patch(p)


        plt.imshow(np_image)
        plt.axis('off')
        plt.show()

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
        print('boxes: ', bboxes )
        # plot_results(im, probas[keep], bboxes_scaled, labels)

    def plot_inference_qa(im, caption):
        # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0).cuda()

        # propagate through the model
        memory_cache = model_qa(img, [caption], encode_and_save=True)
        outputs = model_qa(img, [caption], encode_and_save=False, memory_cache=memory_cache)

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
        # plot_results(im, probas[keep], bboxes_scaled, labels)

        # Classify the question type
        type_conf, type_pred = outputs["pred_answer_type"].softmax(-1).max(-1)
        ans_type = type_pred.item()
        types = ["obj", "attr", "rel", "global", "cat"]

        ans_conf, ans = outputs[f"pred_answer_{types[ans_type]}"][0].softmax(-1).max(-1)
        global answer
        answer = id2answerbytype[f"answer_{types[ans_type]}"][ans.item()]
        print(f"Predicted answer: {answer}\t confidence={round(100 * type_conf.item() * ans_conf.item(), 2)}")

def clearCacheMDETR():
    torch.cuda.empty_cache()

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
    model.load_state_dict(torch.load('C:/Users/Doravan/Desktop/Hidrolatina/torchtest/Yet-Another-EfficientDet-Pytorch/efficientdet-d2_65_9200.pth'))
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

        
        threshold = 0.9
        iou_threshold = 0.4

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

        ##FILTER EMPTY DETECTIONS
        bads=[]
        for i in range((out[0]['scores']).size):
            detected_boxes= out[0]['rois'][i]
            if detected_boxes[0]==detected_boxes[2]:
                bads.append(i)
        out[0]['rois']=np.delete(out[0]['rois'],bads,axis=0)
        out[0]['scores']=np.delete(out[0]['scores'],bads,axis=0)
        out[0]['class_ids']=np.delete(out[0]['class_ids'],bads,axis=0)

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
                global date_hour
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

            global im
            im = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            print(im)
            # imgplot = plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            # plt.show()
            # print(imgplot)
            # print('ymin',ymin)
            # print('fdsfssdfdsf')
            # print('detected_boxes',detected_boxes)
            # print('cropped_img',cropped_img)
            # if cropped_img.size == 0:
            #     continue
            # else:
            #     #cv2.imwrite('cropped_image_{}.jpg'.format(i), cropped_img)
            #     print('cropped_img')
            # imagencamera = Image.fromarray(cropped_img, 'RGB')
    ################################Borrar###########################################
    cropPerson =  imageClip(ImageTk.PhotoImage(im), 'Person detected on '+date_hour)
    listImagenClip.append(cropPerson)


def MDETR():
    import cv2
    # im = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    # im = Image.fromarray(imagencamera)
    plot_inference(im, "a hand")
    im_hand=im.crop((bboxes[len(bboxes)-1]))
    plot_inference(im, "a tiny head")
    global im_head
    im_head=im.crop((bboxes[len(bboxes)-1]))
    # cropPersonHead = imageClip(im_head, "")
    # listImagenClip.append(cropPersonHead)
    # im_head.save('clip/cropped_head.jpg')
    plot_inference_qa(im_hand, "what color are the fingers?")
    if answer =='purple' or answer =='blue':
        gloves= 'Yes'

    else:
        gloves= 'No'
    cropPersonHand = imageClip(ImageTk.PhotoImage(im_hand), 'Gloves: '+gloves)
    listImagenClip.append(cropPersonHand)

    global objectListMDETR
    objectListMDETR= im_head

def loadClip():
    import clip
    import glob

    global class_names, candidate_captions
    class_names=['mask and headphones', 'mask and goggles ','mask, headphones and goggles','mask', 'goggles', 'no_mask']
    candidate_captions=['A head with a medical mask and headphones',
    'A head with goggles and medical mask',
    'A head with a mask and goggles and headphones',
    'A face with medical mask',
    'A face with goggles',
    'Just a face']

    global argmax
    def argmax(iterable):
        return max(enumerate(iterable), key=lambda x: x[1])[0]

    ##################Arreglar global##################
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    ##################Arreglar global##################
    global modelc, process
    modelc, process = clip.load("ViT-B/32", device=device)

    ##################Arreglar global##################
    global text
    text = clip.tokenize(candidate_captions).to(device)

def clip():
    # head=Image.open('E:/Softmaking/Proyectos/Hidrolatina/valorant.jpg')
    image = process(im_head).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = modelc.encode_image(image)
        text_features = modelc.encode_text(text)
        
        logits_per_image, logits_per_text = modelc(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        pred = class_names[argmax(list(probs)[0])]
        print(pred)
        
    cropPerson = imageClip(ImageTk.PhotoImage(im_head), 'PPE in head: '+pred)
    listImagenClip.append(cropPerson)
    #     np_image = np.array(head)
    # plt.imshow(np_image)

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
app_width = 300
app_height = 250

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x = (screen_width/2) - (app_width/2)
y = (screen_height/2) - (app_height/2)

root.geometry(f'{app_width}x{app_height}+{int(x)}+{int(y)}')



#Def
def popup(message):
    messagebox.showinfo(message=message)

def folderSelect():
    folder_selected = filedialog.askdirectory()
    print(folder_selected)

def checkListImagenClip():
    if len(listImagenClip) == 0 :
        popup('No hay datos que mostrar')
    else:
        showImageClipTk


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

    clicked = StringVar()
    clicked.set(option[0])
    drop = OptionMenu(downloadModelsTk, clicked, *option )
    drop.pack()

    pri = Button(downloadModelsTk, text='Descargar modelos', command=download)
    pri.pack()

    #Buttons

    closeWindow = Button(downloadModelsTk, text="Cerrar Ventana", command=closeTk)
    closeWindow.pack()

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
    global labelText
    global listImagenClip

    

    #Def into tk
    def closeTk():
        #Destroy window
        imageClipTk.destroy()
        root.deiconify()

    def forward(imageNumber):
        #Global Declarations into Def tk
        global labelimage
        global buttonBack
        global buttonForward
        global labelText
        global listImagenClip

        #Image
        labelimage.grid_forget()
        labelimage = Label(imageClipTk, image=listImagenClip[imageNumber].getImage())
        buttonForward = Button(imageClipTk, text='>>', command=lambda:forward(imageNumber+1))
        buttonBack = Button(imageClipTk, text='<<', command=lambda:back(imageNumber-1))

        #Text
        labelText.grid_forget()
        labelText = Label(imageClipTk, text=listImagenClip[imageNumber].getAnswer())

        if imageNumber == len(listImagenClip)-1:
            buttonForward = Button(imageClipTk, text='>>', state=DISABLED)
        
        labelimage.grid(row=0, column=0, columnspan=3)
        buttonBack.grid(row=2, column=0)
        buttonForward.grid(row=2, column=2)

        labelText.grid(row=1, column=1)

        return

    def back(imageNumber):
        #Global Declarations into Def tk
        global labelimage
        global buttonBack
        global buttonForward
        global labelText
        global listImagenClip

        #Image
        labelimage.grid_forget()
        labelimage = Label(imageClipTk, image=listImagenClip[imageNumber].getImage())
        buttonForward = Button(imageClipTk, text='>>', command=lambda:forward(imageNumber+1))
        buttonBack = Button(imageClipTk, text='<<', command=lambda:back(imageNumber-1))

        #Text
        labelText.grid_forget()
        labelText = Label(imageClipTk, text=listImagenClip[imageNumber].getAnswer())

        if imageNumber == 0:
            buttonBack = Button(imageClipTk, text='<<', state=DISABLED)
        
        labelimage.grid(row=0, column=0, columnspan=3)
        buttonBack.grid(row=2, column=0)
        buttonForward.grid(row=2, column=2)

        labelText.grid(row=1, column=1)
        
        return

    #Hide Root Window
    root.withdraw()

    #Label Tk
    labelimage = Label(imageClipTk, image=listImagenClip[0].getImage())
    labelimage.grid(row=0, column=0, columnspan=3)
    labelText = Label(imageClipTk, text=listImagenClip[0].getAnswer())
    labelText.grid(row=1, column=1)

    #Buttons Tk
    buttonBack = Button(imageClipTk, text='<<', command=lambda:back, state=DISABLED)
    buttonBack.grid(row=2, column=0)

    exitButton = Button(imageClipTk, text="Cerrar ventana", command=closeTk)
    exitButton.grid(row=2, column=1)

    if len(listImagenClip) == 1:
        buttonForward = Button(imageClipTk, text='>>', command=lambda:forward(1), state=DISABLED)
        buttonForward.grid(row=2, column=2)
    else:
        buttonForward = Button(imageClipTk, text='>>', command=lambda:forward(1))
        buttonForward.grid(row=2, column=2)


def openConfigurationTk():
    global imagen
    # Config tk
    configurationTk = Toplevel()
    configurationTk.resizable(False,False)
    configurationTk.protocol("WM_DELETE_WINDOW", exit)
    configurationTk.title("Configuraciones")
    # configurationTk.overrideredirect(True)

    imagen = ImageTk.PhotoImage(Image.open("E:/Softmaking/Proyectos/Hidrolatina/valorant.jpg"))

    width = 200
    height = 300
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width/2) - (app_width/2)
    y = (screen_height/2) - (app_height/2)

    configurationTk.geometry(f'{width}x{height}+{int(x)}+{int(y)}')

    #Labels Tk
    # labelimagen = Label(configurationTk, image=imagen)
    # labelimagen.pack()

    labelTest = Label(configurationTk, text='fdgdf')
    labelTest.pack()

    #Buttons Tk
    buttonDirectory = Button(configurationTk, text="Cambiar directorio", command=folderSelect)
    buttonDirectory.pack()

    buttonThreshold = Button(configurationTk, text="Cambiar Threshold")
    buttonThreshold.pack()

    buttonIou_threshold = Button(configurationTk, text="Cambiar Iou Threshold")
    buttonIou_threshold.pack()

    buttonDetLimit = Button(configurationTk, text="Cambiar limite de capturas")
    buttonDetLimit.pack()

    buttonClass = Button(configurationTk, text="Cambiar clases")
    buttonClass.pack()

    closeWindow = Button(configurationTk, text="Cerrar Ventana", command=lambda:configurationTk.destroy())
    closeWindow.pack()

#Buttons
# downloadModels = Button(root, text="Modelos", command=openDownloadModelsTk).pack()

clearMDETRyButton = Button(root, text='Limpiar Cache', command=clearCacheMDETR).pack()
importLibraryButton = Button(root, text='Cargar MDETR', command=importMDETER).pack()
loadClipButton = Button(root, text='Cargar Clip', command=loadClip).pack()
loadEfficientIButton = Button(root, text='Cargar EfficientDet', command=loadEfficient).pack()
pytorchCameraButton = Button(root, text='Evaluación de Camara', command=pytorchCamera).pack()
MDETRButton = Button(root, text='MDETR', command=MDETR).pack()
clipButton = Button(root, text='Clip', command=clip).pack()
showImageClipButton = Button(root, text='Resultados', command=checkListImagenClip).pack()
configButton = Button(root, text="Configuraciones", command=openConfigurationTk, fg="blue").pack()


exitButton = Button(root, text="Salir", command=root.quit)
exitButton.pack()

root.mainloop()