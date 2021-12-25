import re
from PIL import Image
from PIL.Image import init
import requests
import pathlib
import json
from collections import defaultdict
import numpy as np

import torch
import torchvision.transforms as T
import clip as clipit


from effdet.utils.inference import init_effdet_model,inference_effdet_model

class PpeDetector:

    def __init__(self):
        self.weigths_effdet = 'https://github.com/EquipoVandV/EfficientDetVandV/blob/main/effdet/logs/person_coco/efficientdet-d2_58_8260_best.pth'
        self.obj_list = ['person']
        # self.effdet_weight='https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d4.pth'
        # self.obj_list= ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        #                     'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        #                 'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
        #                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        #                 'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
        #                 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        #                 'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
        #                 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        #                 'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        #                 'toothbrush']
        self.names_ppe = {'im_head': ['Casco', 'Audífonos', 'Antiparras', 'Mascarilla'], 'im_hand': ['Guantes'], 'im_boot': ['Botas']}
        self.candidate_captions={'im_head': [['A white hat','A head'], ['a big headset', 'a head'],['A face with glasses', 'A head'],['Head with a medical mask', 'Just a head']],
                    'im_hand':[['A blue hand', 'A pink hand']],
                    'im_boot':[['A large boot', 'A small shoe']]}
        self.importMdetr = self.importMDETR()
    
    def argmax(self, iterable):
            return max(enumerate(iterable), key=lambda x: x[1])[0]

    class importMDETR:

        # def __init__(self):
           
            # torch.set_grad_enabled(False);
            # temp = pathlib.PosixPath
            # pathlib.PosixPath = pathlib.WindowsPath
            # self.init()
            # model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=True, return_postprocessor=True)
            # self.model = model.cuda()
            # self.model.eval();

            # # global transform, box_cxcywh_to_xyxy, rescale_bboxes, COLORS, plot_results, id2answerbytype, plot_inference, plot_inference_qa
            # # standard PyTorch mean-std input image normalization
            # self.transform = T.Compose([
            #     T.Resize(800),
            #     T.ToTensor(),
            #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # ])
            # print("MDETR cargado")

        def init(self):
            torch.set_grad_enabled(False);
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=True, return_postprocessor=True)
            self.model = model.cuda()
            self.model.eval();

            self.transform = T.Compose([
                T.Resize(800),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            # out=self.plot_inference(Image.fromarray(np.zeros((10,10,3), np.uint8)), 'empty')
            # del out
            torch.cuda.empty_cache()
            print("MDETR cargado")
            return self.model, self.transform


        # for output bounding box post-processing
        def box_cxcywh_to_xyxy(self, x):
            x_c, y_c, w, h = x.unbind(1)
            b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
                (x_c + 0.5 * w), (y_c + 0.5 * h)]
            return torch.stack(b, dim=1)

        def rescale_bboxes(self, out_bbox, size):
            img_w, img_h = size
            b = self.box_cxcywh_to_xyxy(out_bbox)
            b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
            return b

        # colors for visualization
        # COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
        #         [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

        # import json
        # answer2id_by_type = json.load(requests.get("https://nyu.box.com/shared/static/j4rnpo8ixn6v0iznno2pim6ffj3jyaj8.json", stream=True).raw)
        # id2answerbytype = {}                                                       
        # for ans_type in answer2id_by_type.keys():                        
        #     curr_reversed_dict = {v: k for k, v in answer2id_by_type[ans_type].items()}
        #     id2answerbytype[ans_type] = curr_reversed_dict 


        def plot_inference(self, im, caption):
        # mean-std normalize the input image (batch-size: 1)
            img = self.transform(im).unsqueeze(0).cuda()

            # propagate through the model
            memory_cache = self.model(img, [caption], encode_and_save=True)
            outputs = self.model(img, [caption], encode_and_save=False, memory_cache=memory_cache)

            # global probas, keep
            # keep only predictions with 0.7+ confidence
            self.probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
            self.keep = (self.probas > 0.7).cpu()

            # convert boxes from [0; 1] to image scales
            bboxes_scaled = PpeDetector.importMDETR().rescale_bboxes(outputs['pred_boxes'].cpu()[0, self.keep], im.size)

            # Extract the text spans predicted by each box
            positive_tokens = (outputs["pred_logits"].cpu()[0, self.keep].softmax(-1) > 0.1).nonzero().tolist()
            predicted_spans = defaultdict(str)
            for tok in positive_tokens:
                item, pos = tok
                if pos < 255:
                    span = memory_cache["tokenized"].token_to_chars(0, pos)
                    predicted_spans [item] += " " + caption[span.start:span.end]
            labels = [predicted_spans [k] for k in sorted(list(predicted_spans .keys()))]
            # global bboxes
            self.bboxes=bboxes_scaled.numpy()
            # print('boxes: ', bboxes)
            # plot_results(im, probas[keep], bboxes_scaled, labels)

            return (self.bboxes[PpeDetector().argmax(self.probas[self.keep])])
        
        def __exit__(self):
            print("MDETR cargado")

    
    def loadEfficientDet(self):
        # self.weigths_effdet='https://github.com/EquipoVandV/EfficientDetVandV/blob/main/effdet/logs/person_coco/efficientdet-d2_58_8260_best.pth'
        # self.weigths_effdet = 'C:/hidrolatina/EfficientDetVandV-main/effdet/logs/person_coco/efficientdet-d2_58_8260_best.pth'
        # self.obj_list = ['person']
   
        self.model_effdet = init_effdet_model(self.weigths_effdet, self.obj_list)
        # out=self.efficientDet(np.zeros((10,10,3), np.uint8))
        # del out
        torch.cuda.empty_cache()

        print('EfficientDET Cargado')
        # return self.model_effdet

    def MDETR(self, im):
        bboxes_body = self.importMdetr.plot_inference( im, "a hand")
        # plot_inference(im, "a hand")
        im_hand=im.crop(bboxes_body)

        bboxes_body = self.importMdetr.plot_inference( im, "a head")
        # plot_inference(im, "a head")
        im_head=im.crop(bboxes_body)
        
        bboxes_body = self.importMdetr.plot_inference( im, "a boot")
        im_boot=im.crop(bboxes_body)

        objectListMDETR= {'im_head':im_head, 'im_hand': im_hand, 'im_boot': im_boot}
        return objectListMDETR

    def loadClip(self):
        # global clipit

        # global candidate_captions
        # self.candidate_captions={'im_head': [['a head with a yellow helmet','Just a head'], ['Head with headphones', 'Just a head'],['a Head with goggles', 'Just a head'],['Head with a medical mask', 'Just a head']],
        #             'im_hand':[['A blue hand', 'A pink hand']],
        #             'im_boot':[['A black boot', 'A shoe']]}
        # candidate_captions={'im_head': [['a white hat','A head'], ['a big headset', 'a face'],['a face with glasses', 'A head'],['Mask', 'Just a head']],
        #             'im_hand':[['A blue hand', 'A pink hand']],
        #             'im_boot':[['a large boot', 'a small shoe']]}

        # self.names_ppe = {'im_head': ['Casco', 'Audífonos', 'Antiparras', 'Mascarilla'], 'im_hand': ['Guantes'], 'im_boot': ['Botas']}

        # global device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "cpu"

        # global modelc, process
        self.modelc, self.process = clipit.load("ViT-B/32", device=self.device)

        # global nstr
        # def nstr(obj):
        #     return [name for name in globals() if globals()[name] is obj][0]
        print('Clip Cargado')

    def clip(self, bodypart, mdetr_list):
        pred_clip=[]
        for i in range(len(self.candidate_captions[bodypart])):
            text = clipit.tokenize(self.candidate_captions[bodypart][i]).to(self.device)
            image = self.process(mdetr_list[bodypart]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # image_features = modelc.encode_image(image)
                # text_features = modelc.encode_text(text)
                
                logits_per_image, logits_per_text = self.modelc(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                # pred = class_names[argmax(list(probs)[0])]
                if self.argmax(list(probs)[0])== 0:
                    pred_clip.append('Ok')
                else:
                    pred_clip.append('No Detectado')

        return pred_clip

    def efficientDet(self, img):
        self.out = inference_effdet_model(self.model_effdet, img, threshold=0.8)
        torch.cuda.empty_cache()
        return self.out

