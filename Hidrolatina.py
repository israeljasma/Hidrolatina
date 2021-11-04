from glob import glob
import queue
from sys import path
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import simpledialog
import cv2
import os
import platform
import time
from multiprocessing import Queue
from threading import Thread

from Services import API_Services
from UserClass import Person
# from varname import varname, nameof

from numpy import CLIP
from imagenClipClass import imageClip
from NFCClass import NFC
from FileManagementClass import FileManagement

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

def downloadEfficientDet():
    url = 'https://github.com/EquipoVandV/VandVEfficientDet/archive/refs/heads/main.zip'
    FileManagement.extractFile(FileManagement.downloadFile(url, DATA_DIR), DATA_DIR)
    return

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
    # model_qa = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5_gqa', pretrained=True, return_postprocessor=False)
    # model_qa = model_qa.cuda()
    # model_qa.eval();
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

    # def plot_inference_qa(im, caption):
    #     # mean-std normalize the input image (batch-size: 1)
    #     img = transform(im).unsqueeze(0).cuda()

    #     # propagate through the model
    #     memory_cache = model_qa(img, [caption], encode_and_save=True)
    #     outputs = model_qa(img, [caption], encode_and_save=False, memory_cache=memory_cache)

    #     # keep only predictions with 0.7+ confidence
    #     probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
    #     keep = (probas > 0.7).cpu()

    #     # convert boxes from [0; 1] to image scales
    #     bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)

    #     # Extract the text spans predicted by each box
    #     positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
    #     predicted_spans = defaultdict(str)
    #     for tok in positive_tokens:
    #         item, pos = tok
    #         if pos < 255:
    #             span = memory_cache["tokenized"].token_to_chars(0, pos)
    #             predicted_spans [item] += " " + caption[span.start:span.end]

    #     labels = [predicted_spans [k] for k in sorted(list(predicted_spans .keys()))]
    #     # plot_results(im, probas[keep], bboxes_scaled, labels)

    #     # Classify the question type
    #     type_conf, type_pred = outputs["pred_answer_type"].softmax(-1).max(-1)
    #     ans_type = type_pred.item()
    #     types = ["obj", "attr", "rel", "global", "cat"]

    #     ans_conf, ans = outputs[f"pred_answer_{types[ans_type]}"][0].softmax(-1).max(-1)
    #     global answer
    #     answer = id2answerbytype[f"answer_{types[ans_type]}"][ans.item()]
    #     print(f"Predicted answer: {answer}\t confidence={round(100 * type_conf.item() * ans_conf.item(), 2)}")
    print("MDETR cargado")

def clearCacheMDETR():
    torch.cuda.empty_cache()

def loadEfficient():
    ### Arreglar directorio
    import sys
    sys.path.append("C:/Users/Doravan/Desktop/Hidrolatina/torchtest/Yet-Another-EfficientDet-Pytorch")
    import torch
    from torch.backends import cudnn
    from threading import Thread, Lock

    from backbone import EfficientDetBackbone
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    from efficientdet.utils import BBoxTransform, ClipBoxes
    from utils.utils import preprocess, invert_affine, postprocess

    global preprocess, invert_affine, postprocess, BBoxTransform, ClipBoxes

    compound_coef = 2
    force_input_size = None  # set None to use default size

    global use_cuda, use_float16
    use_cuda = True
    use_float16 = False                                                 
    cudnn.fastest = True
    cudnn.benchmark = True

    global obj_list
    obj_list = ['person']

    global input_size

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    global model_ed

    model_ed = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),

                                # replace this part with your project's anchor config
                                ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                                scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    model_ed.load_state_dict(torch.load('C:/Users/Doravan/Desktop/Hidrolatina/torchtest/Yet-Another-EfficientDet-Pytorch/efficientdet-d2_65_9200.pth'))
    # model_ed.load_state_dict(torch.load('E:/Users/darkb/OneDrive/Documentos/EIE/Tesis/Pruebas_de_codigos/Yet-Another-EfficientDet-Pytorch/weights/efficientdet-d3_206_34776_best.pth'))
    model_ed.requires_grad_(False)
    model_ed.eval()

    if use_cuda:
        model_ed = model_ed.cuda()
    if use_float16:
        model_ed = model_ed.half()

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

def pytorchCamera():
    ###REAL##
    import cv2
    cap = cv2.VideoCapture(0)

    import numpy as np
    import datetime
    import matplotlib.pyplot as plt

    det=0

    while True:
        # Read frame from camera
        cap.set(cv2.CAP_PROP_FPS,16)
        ret, image_np = cap.read()
        image_path=[image_np]

        
        threshold = 0.6
        iou_threshold = 0.1

        # # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # image_np_expanded = np.expand_dims(image_np, axis=0)
        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_size)

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        with torch.no_grad():
            features, regression, classification, anchors = model_ed(x)

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
            if det==20:     #break in det-1
                global date_hour
                now = datetime.datetime.now()
                date_hour='%d/%d/%d-%d:%d:%d'%( now.day, now.month, now.year, now.hour, now.minute, now.second )
                print(date_hour)
                cap.release()
                cv2.destroyAllWindows()
                break

        if len(out[0]['scores'])==0:
            det=0


        
        # Save Bounding Boxes

        for i in range((out[0]['scores']).size):
            detected_boxes= out[0]['rois'][i]
              


        # Crop and save detedtec bounding box image

            xmin = int((detected_boxes[0]))
            ymin = int((detected_boxes[1]))
            xmax = int((detected_boxes[2]))
            ymax = int((detected_boxes[3]))
            cropped_img = image_np[ymin:ymax,xmin:xmax]

            if cropped_img.size != 0:
                imagencamera = cropped_img
                global im
                im = Image.fromarray(cv2.cvtColor(imagencamera, cv2.COLOR_BGR2RGB))
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


def MDETR(im):
    import cv2
    # im = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    # im = Image.fromarray(imagencamera)
    plot_inference(im, "a hand")
    im_hand=im.crop((bboxes[argmax(probas[keep])]))
    plot_inference(im, "a tiny head")
    global im_head
    im_head=im.crop((bboxes[argmax(probas[keep])]))
    plot_inference(im, "a rain boot")
    global im_boot
    im_boot=im.crop((bboxes[argmax(probas[keep])]))
    # cropPersonHead = imageClip(im_head, "")
    # listImagenClip.append(cropPersonHead)
    # im_head.save('clip/cropped_head.jpg')
    # plot_inference_qa(im_hand, "what color are the fingers?")
    # if answer =='purple' or answer =='blue':
    #     gloves= 'Yes'

    # else:
    #     gloves= 'No'
    # cropPersonHand = imageClip(ImageTk.PhotoImage(im_hand), 'Gloves: '+gloves)
    # listImagenClip.append(cropPersonHand)

    objectListMDETR= {'im_head':im_head, 'im_hand': im_hand, 'im_boot': im_boot}
    return objectListMDETR

def loadClip():
    global clipit
    import clip as clipit
    import glob

    global candidate_captions
    # class_names={'im_head': [['helmet','no_helmet'], ['mask', 'no_mask'], ['goggles', 'no_goggles'], ['headphones', 'no_headphones']],
    #             'im_hand':[['gloves', 'no_gloves']],
    #             'im_boot': [['boots', 'no boots']]}
    candidate_captions={'im_head': [['a white hat','A head'], ['big black headphone', 'A head'],['head with glasses', 'A face'],['Mask', 'Just a head']],
                'im_hand':[['A blue hand', 'A pink hand']],
                'im_boot':[['a large boot', 'a small shoe']]}

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

def clip(bodypart):
    # head=Image.open('E:/Softmaking/Proyectos/Hidrolatina/valorant.jpg')
    pred_clip=[]
    for i in range(len(candidate_captions[bodypart])):
        # head=Image.open('E:/Users/darkb/OneDrive/Documentos/EIE/Tesis/Pruebas_de_codigos/Bases_de_datos/Implementos seguridad/others/goggles_headphones/head (99).jpg')
        # print(candidate_captions[nstr(bodypart)][i])
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
        #     np_image = np.array(head)
        # plt.imshow(np_image)
    return pred_clip

def loadALL():
    # from threading import Thread
    # thread= Thread(target=librerias, args=())
    # thread.start()

    importMDETER()
    loadClip()
    loadEfficient()
    messagebox.showinfo(message="Dependencias cargadas")

########Windows#######

# root = Tk()
# root.title("Softmaking")
# root.resizable(False,False)
# root.config(background="#cceeff")
# root.resizable(False,False)
# root.overrideredirect(True)
# root.geometry(f'{root.winfo_screenwidth()}x{root.winfo_screenheight()}')

# #Frame
# rootFrame = Frame(root, width=round(root.winfo_screenwidth()), height=round(root.winfo_screenheight()), bg='#cceeff')
# rootFrame.grid()

# topFrame = Frame(rootFrame, width=round(rootFrame.winfo_reqwidth()), height=rootFrame.winfo_reqheight()*0.4, bg='#cceeff')
# topFrame.grid(row=0, column=0)

# bottomFrame = Frame(rootFrame, width=round(rootFrame.winfo_reqwidth()), height=rootFrame.winfo_reqheight()*0.6, bg='#cceeff')
# bottomFrame.grid(row=1, column=0)

# imageLogoRoot = Image.open('images/logo_hidrolatina.png')
# imageLogoRoot = imageLogoRoot.resize((round(topFrame.winfo_reqwidth()), round(topFrame.winfo_reqheight())), Image.ANTIALIAS)
# imageLogoRoot = ImageTk.PhotoImage(imageLogoRoot)

# imageLabelLeft_Frame = Label(topFrame, image=imageLogoRoot, bg='#cceeff', borderwidth=0)
# imageLabelLeft_Frame.grid(row=0, column=0)

#Center windows
# app_width = 300
# app_height = 300

# screen_width = root.winfo_screenwidth()
# screen_height = root.winfo_screenheight()

# x = (screen_width/2) - (app_width/2)
# y = (screen_height/2) - (app_height/2)

# root.geometry(f'{app_width}x{app_height}+{int(x)}+{int(y)}')

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

def showPytorchCameraTk():
    #Import
    import matplotlib.pyplot as plt
    import datetime
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
    pytorchCameraTk.resizable(False,False)
    pytorchCameraTk.config(background="#cceeff")
    # pytorchCameraTk.overrideredirect(True)
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
    
    #Slider window (slider controls stage position)
    # sliderFrame = Frame(pytorchCameraTk, width=600, height=100)
    # sliderFrame.grid(row = 600, column=0, padx=10, pady=2)

    #Capture video frames
    labelVideo = Label(cameraFrame)
    labelVideo.grid(row=0, column=0)
    # cap = CameraStream(varCamera).start()
    cap = cv2.VideoCapture(0)

    camWidth = round(cameraFrame.winfo_reqwidth())
    camHeight = round(cameraFrame.winfo_reqheight()*0.85)

    #Def into tk
    def closeTk():
        #Destroy window
        cap.release()
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
        frame = cap.read()

        threshold = 0.4
        iou_threshold = 0.1

        image_path= [frame]
        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_size)

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        with torch.no_grad():
            features, regression, classification, anchors = model_ed(x)

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

        cv2image = cv2.cvtColor(cv2.resize(ori_img, (600, 500)), cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        labelVideo.imgtk = imgtk
        labelVideo.configure(image=imgtk)

        global det
        print(det)
        if len(out[0]['class_ids']) == 0:
            det = 0
        if len(out[0]['class_ids']) > 0:
            det += 1
            if det==20:

                print("Reset")
                for i in range((out[0]['scores']).size):
                    detected_boxes= out[0]['rois'][i]

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
                global mdetr_list
                mdetr_list=MDETR(im)
################################################ CORRERGIR ###############################################
################################################ CORRERGIR ###############################################
################################################ CORRERGIR ###############################################
                # print(mdetr_list)
                global listImagenClip
                for bodypart in mdetr_list.keys():
                    listImagenClip.append(imageClip(ImageTk.PhotoImage(mdetr_list[bodypart].resize((150,150))), clip(bodypart)))
                    # print(bodypart)
                
                updateLabel()
################################################ CORRERGIR ###############################################
################################################ CORRERGIR ###############################################
################################################ CORRERGIR ###############################################
                
        if det<20:
            labelVideo.after(10, showFrame)
    
    def timePop(booleanAnswer):
        ContinueExecuting = True
        starting_point = time.time()
        while ContinueExecuting:
            elapsed_time = time.time () - starting_point
            elapsed_time_int = int(elapsed_time)
            if elapsed_time_int >= 10:
                popupIdentificationTk(booleanAnswer)
                ContinueExecuting = False

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
        for i in range(len(listImagenClip)):
            for j in range(len(listImagenClip[i])):
                if listImagenClip[i].getAnswer()[j] == 'OK':
                    booleanAnswer = True
                else:
                    booleanAnswer = False
                    break

        thread = Thread(target=timePop,args=(booleanAnswer,))
        thread.start()

    testFrame()
    # showFrame()

    exitButton = Button(pytorchCameraTk, text='Cerrar ventana', command=closeTk)
    exitButton.grid(row=1, column=0)

    testButtonUpdate = Button(pytorchCameraTk, text='Test Update', command=updateLabel)
    testButtonUpdate.grid(row=1, column=1)

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

    x = (screen_width/2) - (app_width/2)
    y = (screen_height/2) - (app_height/2)

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

def identify():
    from smartcard.CardRequest import CardRequest
    from smartcard.Exceptions import CardRequestTimeoutException
    from smartcard.CardType import AnyCardType
    from smartcard import util
    import time
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
            status = util.toHexString([sw1, sw2])
            if uid == "44 CE 4A 0B":

            # print the ATR and UID of the card
            # print("ATR = {}".format(util.toHexString(conn.getATR())))
                print("Operador Reconocido")
                break
        except:
            pass
    time.sleep(2)
    # showPytorchCameraTk()

def nfc_identifyTk():
    # Config tk
    NFC_Tk = Toplevel()
    NFC_Tk.resizable(False,False)
    NFC_Tk.title("Identificación")
    NFC_Tk.overrideredirect(True)
    NFC_Tk.geometry(f'{NFC_Tk.winfo_screenwidth()}x{NFC_Tk.winfo_screenheight()}')

    #Def
    def time_string():
        return time.strftime('%H:%M:%S')

    def update():
        timeLabel.configure(text=time_string())
        # Recursive
        timeLabel.after(1000, update)

    def thread_identify():
        while True:
            NFC.testnfc()
        # thread = Thread(target=NFC.identify, args=(q,))
        # thread.start()
        return True

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

    #Buttons Tk
    # Button(NFC_Tk, text="Cerrar Ventana", command=lambda:closeTk()).pack(pady=10)

    # thread= Thread(target=NFC.identify, args=())
    # thread.start()

    # thread.join()

def popupIdentificationTk(booleanAnswer=False):
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

def openConfigurationTk():
    # Config tk
    configurationTk = Toplevel()
    configurationTk.resizable(False,False)
    configurationTk.protocol("WM_DELETE_WINDOW", exit)
    configurationTk.title("Configuraciones")
    # configurationTk.overrideredirect(True)

    width = 200
    height = 300
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width/2) - (app_width/2)
    y = (screen_height/2) - (app_height/2)

    configurationTk.geometry(f'{width}x{height}+{int(x)}+{int(y)}')

    #Def
    def closeTk():
        configurationTk.destroy()
        root.deiconify()

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
    root.withdraw()

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

    closeWindow = Button(configurationTk, text="Cerrar Ventana", command=lambda:closeTk())
    closeWindow.pack()

#Buttons
# clearMDETRyButton = Button(bottomFrame, text='Limpiar Cache', command=clearCacheMDETR).grid()
# loadALLButton=  Button(bottomFrame, text='Cargar Dependencias', command=loadALL).grid()
# # # MDETRButton = Button(root, text='MDETR', command=MDETR).pack()
# # # clipButton = Button(root, text='Clip', command=clip).pack()
# # # showImageClipButton = Button(root, text='Resultados', command=checkListImagenClip).pack()
# configButton = Button(bottomFrame, text='Configuraciones', command=openConfigurationTk, fg='blue').grid()

# testButton = Button(bottomFrame, text='Test Camara',command=showPytorchCameraTk, fg='red').grid()
# testButton = Button(bottomFrame, text='Test download',command=downloadEfficientDet, fg='red').grid()
# testButton = Button(bottomFrame, text='Test NFC',command=nfc_identifyTk, fg='red').grid()
# testButton = Button(bottomFrame, text='Test POPUP',command=popupIdentificationTk, fg='red').grid()

# exitImageButton = Image.open('images/exit2.png')
# # exitImageButton = imageLogoRoot.resize((round(topFrame.winfo_reqwidth()), round(topFrame.winfo_reqheight())), Image.ANTIALIAS)
# exitImageButton = ImageTk.PhotoImage(exitImageButton)

# exitButton = Button(bottomFrame, text="Salir", image=exitImageButton, command=root.quit, bg='#cceeff', activebackground='#cceeff', borderwidth=0)
# exitButton.grid()

# root.mainloop()

############ Start App ############
root = Tk()
root.geometry('350x500+500+50')
root.resizable(0,0)
root.config(bg='#CCEEFF')
root.title('Hidrolatina')

#Def
def verification():
    user = userEntry.get()
    password = passwordEntry.get()
    person = API_Services.login(user, password)
    if 'token' in person:
        user = Person(person['user']['name'], person['user']['last_name'], person['user']['email'], person['token'])
        messagebox.showinfo(message=[person['user']['name'], person['user']['last_name']], title="Login")
    else:
        messagebox.showinfo(message=person['error'], title="Login")

def closeLogin():
    root.destroy()
    root.quit()

logo = Image.open('images/logo_hidrolatina.png')
logo = logo.resize((325, 97), Image.ANTIALIAS)
logo = ImageTk.PhotoImage(logo)
logoLabel = Label(root, image=logo, width=325, height=97, bg='#CCEEFF')
logoLabel.pack(pady=30)

userLabel = Label(root, text='Usuario', bg='#CCEEFF').pack()
userEntry = Entry()
userEntry.pack()

passwordLabel = Label(root, text='Contraseña', bg='#CCEEFF').pack()
passwordEntry = Entry(show='*')
passwordEntry.pack()

loginButton = Button(root, command=verification, text='Iniciar Sesión', bg='#c2eaff').pack()

closeButton = Button(root, text='Salir', command=closeLogin, bg='#c2eaff').pack()
root.mainloop()