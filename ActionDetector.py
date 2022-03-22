from datetime import datetime
from tkinter.constants import FALSE
import cv2
import numpy as np
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,inference_bottom_up_pose_model,
                         vis_pose_result, process_mmdet_results, vis_3d_pose_result, inference_pose_lifter_model, extract_pose_sequence, get_track_id,vis_pose_tracking_result)
import pandas as pd
from mmpose.core import SimpleCamera
from mmpose.apis.inference import init_pose_model
from mmaction.apis import inference_recognizer, init_recognizer
from numpy.core.shape_base import block
import torch
from torch._C import Block
from torch.nn.functional import threshold
from xtcocotools.coco import COCO
from effdet.utils.inference import init_effdet_model, inference_effdet_model
from CameraStream import CameraStream
import tkinter
from tkinter import filedialog



from PIL import Image, ImageTk
from sklearn.cluster import KMeans

# pose_config = 'C:/Users/Hidrolatina/Downloads/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
# pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'

# effdet_weight= 'C:/hidrolatina/EfficientDetVandV/effdet/logs/person_coco/efficientdet-d2_58_8260_best.pth'
# effdet_weight='https://github.com/EquipoVandV/EfficientDetVandV/blob/main/effdet/logs/person_coco/efficientdet-d2_58_8260_best.pth'
# obj_list = ['person']
OMP_NUM_THREADS=1
class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    
    def __init__(self, image, clusters=1):
        self.CLUSTERS = clusters
        self.IMAGE = image
        
    def dominantColors(self):
        img = cv2.cvtColor(self.IMAGE, cv2.COLOR_BGR2RGB)
                
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        #save image after operations
        self.IMAGE = img
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        torch.cuda.empty_cache()
        kmeans.fit(img)
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        #save labels
        self.LABELS = kmeans.labels_
        #returning after converting to integer from float
        return self.COLORS.astype(int)

class ActionDetector:
    def __init__(self):
        self.effdet_weight='https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d4.pth'
        self.obj_list= ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                        'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                        'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                        'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                        'toothbrush']

        self.pose_config = 'C:/Users/Hidrolatina/Downloads/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_384x288_udp.py'
        self.pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w32_coco_384x288_udp-e97c1a0f_20210223.pth'

        self.posec3d_config='C:/Hidrolatina/mmaction2/configs/skeleton/posec3d/5HL_4.py'
        self.posec3d_checkpoint='C:/Hidrolatina/mmaction2/Train/work_dirs/posec3d/5HL_4/latest.pth'

    def load_effdet(self):
        #initialize effdet model
        self.effdet_model= init_effdet_model(self.effdet_weight,self.obj_list, coef=4)

    def load_pose(self):
        # initialize pose model
        self.pose_model = init_pose_model(self.pose_config, self.pose_checkpoint)

    def load_action(self):
        # initialize detector
        self.posec3d_model=init_recognizer(self.posec3d_config,self.posec3d_checkpoint, 'cuda:0')

    def load_zone(self):
        x_cabinet=np.array(list(range(1170,1280)))
        x_grating1=np.array(list(range(1020,1240)))
        x_grating2=np.array(list(range(1115,1260)))
        x_ground= np.array(list(range(900,1300)))
        y_ground_grat=np.array(list(range(100,200)))
        y_ground_grat=np.flip(y_ground_grat)
        y_ground_gab=np.array(list(range(100,200)))
        y_ground_gab=np.flip(y_ground_gab)
        #Generate y coordinates
        ramp_cabinet= lambda t: round(-0.95*t + 1470)
        ramp_grating1= lambda t: round(-0.93*t + 1510)
        ramp_grating2= lambda t: round(-1.2825*t + 2280)
        ramp_ground= lambda t: round(0.3906*t + 345.0539)
        ramp_ground_y_grat= lambda t: round((t- 2450)/-1.25)
        ramp_ground_2_grat= lambda t: round(0.12*t + 650)
        ramp_ground_y_gab= lambda t: round((t- 2680)/-1.6)
        ramp_ground_2_gab= lambda t: round(0.05*t + 440)

        def create_zone(ramp, x,largo=10):
            y = np.array([ramp(xi) for xi in x])
            x_copy=y_copy= np.array([]).astype(int)
            for i in range(largo):
                x_copy=np.append(x_copy,x)
                y_copy=np.append(y_copy,y+i)
            return np.column_stack((x_copy, y_copy))

        def create_zone2(ramp, y,ramp2,ancho=100, largo=500):
            x_copy=y_copy=x_x_copy=y_y_copy=np.array([]).astype(int)
            for i in range(ancho):
                if i==0:
                    x = np.array([ramp(xi) for xi in y])     
                    x_copy=np.append(x_copy,x+i)
                    y_copy=np.append(y_copy,y)

                if i>0:
                    xlim=x_copy[-1]
                    y=np.array(list(range(ramp2(xlim), largo+ramp2(i)) ))
                    x = np.array([ramp(xi) for xi in y])
                    x_copy=np.append(x_copy,x+i)
                    y_copy=np.append(y_copy,y )
            return np.column_stack((x_copy, y_copy))

        self.zona={ 'rejilla_up': create_zone(ramp_grating1, x_grating1, 60), 'rejilla_down': create_zone(ramp_grating2, x_grating2, 90), 'primer_gabinete': create_zone(ramp_cabinet,x_cabinet,60),
                    'suelo_rejilla':create_zone2(ramp_ground_y_grat,y_ground_grat, ramp_ground_2_grat,400, 480),
                    'suelo_gabinete':create_zone2(ramp_ground_y_gab,y_ground_gab, ramp_ground_2_gab,200, 220)}

    def proc_paral(self, queue_anno,queue_action, flag_posec3d_init): 

        posec3d_model=init_recognizer(self.posec3d_config, self.posec3d_checkpoint, 'cuda:0')
        # # model.to('cuda:0')
        fake_anno={'frame_dir': '', 'label': -1, 'img_shape': (10, 10), 'original_shape': (10, 10), 'start_index': 0, 'modality': 'Pose', 'total_frames': 1, 'keypoint': np.zeros((0, 1, 17, 2), dtype=np.float16), 'keypoint_score': np.zeros((0, 1, 17), dtype=np.float16)}
        actions=inference_recognizer(posec3d_model,fake_anno)
        print('Inicializó Modelo Acciones')
        del actions
        torch.cuda.empty_cache()
        flag_posec3d_init.put(True)

        while True:
    

            if not queue_anno.empty():

                anno=queue_anno.get()
                print('Annon Length: ',len(anno['keypoint'][0]))
                print('Starting inference...')
                actions=inference_recognizer(posec3d_model,anno)
                print(actions)
                queue_action.put(actions)
                del actions 
                torch.cuda.empty_cache()
    

                # print(actions)
    def WriteFrame(self, tableview, df):

        tableview["column"] = list(df.columns)
        tableview["show"] = "headings"
        for column in tableview["columns"]:
            tableview.heading(column, text=column)
            tableview.column(column, minwidth=0, width=150, stretch=tkinter.NO)

        tableview.delete(*tableview.get_children())
        df_rows = df.to_numpy().tolist() # turns the dataframe into a list of lists
        for row in df_rows:
          tableview.insert("", "end", values=row)

    # def DownloadpdTk(self):
    #     out = filedialog.asksaveasfilename(defaultextension=".xlsx")
    #     print('out ', out)
    #     self.df.to_excel(out, index=False)
    #     print(self.df)

    def inferenceActionDetector(self, queue_anno, queue_action, labelVideo, showActions, tableview, btaudio):
        just_bboxarea_track=False
        track_flag=True
        wait_for_operator=5
        operator_counter=wait_for_operator


        next_id =next_id_last= 2

        pose_results=pose_results_last = pose_results_list= pose_frames=[]
        flag_main_init=False

        window=5
        
        wait_for_action=20
        counter=wait_for_action
        Partida_r_down=Partida_r_up=Partida_zona_g_1=Partida_g_1=ACCION_WARN= False

        action_name=['Rejilla cerrada','Accion aleatoria','Rejilla abierta','Bombas activadas', 'Gabinete abierto']
        result_actions=[0,0,0]
        
        
        df_data={'Op. Presente':['No'], 'Accion':['No'], 'Riesgo':['No'], 'Hora':[datetime.today().time().isoformat('seconds')], 'Fecha':[datetime.today().date()]}
        self.df = pd.DataFrame(df_data)
        self.WriteFrame(tableview, self.df)
        old_action= old_op_present= op_present ='No'
        actual_action={'name':'No', 'score': ''}
        risk='No'
        
        torch.cuda.empty_cache()
        print('Camara de planta es esta: ', self.varCamera) 
        try:
            self.cam = CameraStream(self.actionvideo_selected,delay=0.03).start()    #test
        # cam = CameraStream('C:/Users/Hidrolatina/Downloads/Videos_dataset/2.mp4',delay=0.03).start()  #test
        # cam = CameraStream('C:/Users/Hidrolatina/Downloads/Videos_dataset/CAM02 acciones 60 ciclos 12-10-2021.wmv', delay=0.03).start()    #t
        # cam = CameraStream(0).start() 
        except AttributeError:
            self.cam=CameraStream(self.varCamera).start()
        while True:
            try:
                img = self.cam.read()
                    # img=cv2.imread('C:/Users/Hidrolatina/Downloads/Dataset/Pose_blue2.png')
                    # img=cv2.imread('C:/Users/Hidrolatina/Downloads/videos_test/capture5.png')
                    # img2=cv2.imread('C:/Users/Hidrolatina/Downloads/Dataset/pose_blue4.png')
                if img is None or self.cam.started is False:
                    ('No hay más frames para mostrar')
                    break
            except Exception as exception:
                print("Exception: {}".format(type(exception).__name__))
                print("Exception message: {}".format(exception))
                break

        
            #EfficientDET person detection
            out=inference_effdet_model(self.effdet_model,img, coef=4, threshold=0.4)  

            p_det=[]
            for j in range(len(out['bbox'])):
                if out['class_ids'][j]==0:                               #Ensure that is person 0 class
                    (x1, y1, x2, y2) = out['bbox'][j].astype(float)
                    score = float(out['scores'][j])

                    p_det_n={'bbox':[x1,y1,x2,y2,score]}
                    p_det.append(p_det_n)

            # # inference detection
            # mmdet_results = inference_detector(det_model, img)

            # # extract person (COCO_ID=1) bounding boxes from the detection results
            # p_det= process_mmdet_results(mmdet_results, cat_id=1)

            if just_bboxarea_track == True:
                try:
                    # Max Area BBOX filter
                    area=0
                    for k in range(len(p_det)):

                        if (p_det[k]['bbox'][2]-p_det[k]['bbox'][0])*(p_det[k]['bbox'][3]-p_det[k]['bbox'][1])>area:
                                area=(p_det[k]['bbox'][2]-p_det[k]['bbox'][0])*(p_det[k]['bbox'][3]-p_det[k]['bbox'][1])
                                p_det[k]['bbox']
                                p_det_max=[p_det[k]]
                        p_det=p_det_max
                except:
                    pass

                
                    
                    

            
            # if len(pose_results_last)<=len(pose_results):
            pose_results, returned_outputs = inference_top_down_pose_model(self.pose_model,
                                                                        img,
                                                                        p_det,
                                                                        # bbox_thr=0.2,
                                                                        format='xyxy',
                                                                        dataset=self.pose_model.cfg.data.test.type)
            ############################ Plot Drawed Zones ###########################################
            # for segmento in self.zona:
            #     if segmento=='suelo_rejilla' or segmento=='suelo_gabinete':
            #         # for point in zona[segmento]:
            #         #     img = cv2.circle(img, (point), radius=1, color=(255, 0, 0), thickness=-1)
            #             pass
            #     else: 
            #         for point in self.zona[segmento]:
            #             img = cv2.circle(img, (point), radius=1, color=(0, 0, 255), thickness=-1)
            # ##################################################################################
            
            # pose_results_list = []
            # pose_results_list.append(copy.deepcopy(pose_results))                       

            ################################ Tracking Operator #################################################
            if track_flag == True:
                pose_results, next_id = get_track_id(pose_results,
                                                pose_results_last,
                                                next_id,

                                                # min_keypoints=0,
                                                use_oks=True,
                                                tracking_thr=0.1,
                                                fps=40,
                                                use_one_euro=True,
                                                )

            
                #Asignar id operador 0
                # try:
                if next_id_last !=next_id  and flag_main_init==True and len(p_det)>0:
                    print('#####Nuevo ID#####')
                    max_area=0
                    for i in range(len(p_det)):
                        detected_boxes= p_det[i]['bbox']
                        head_point= (pose_results[i]['keypoints'][1] + pose_results[i]['keypoints'][2]) / 2
                        xmin = int((detected_boxes[0]))
                        ymin = int((detected_boxes[1]))
                        xmax = int((detected_boxes[2]))
                        # ymax = int((detected_boxes[3]))
                        ymax = (int(head_point[1]))
                        rad=((ymax-ymin)/2)
                        # print('Radius: ', rad)
                        try:
                        # cropped_img = img[int(ymin+3*rad):int(ymax-3*rad),(int(head_point[0]-rad)):(int(head_point[0]+rad))]
                            # cropped_img = img[int(ymin+rad):int(ymax-4*rad),(int(head_point[0]+6)):(int(head_point[0]+9))] 
                            cropped_img = img[int(ymin+rad*.3):int(ymax-rad),(int(head_point[0]+9)):(int(head_point[0]+20))] 
                        
                        
                        # print(len(cropped_img))
                        # im = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                        # pred_clip=clip_it(im, candidate_captions)
                        # if pred_clip==['OK']:
                        #     print(pred_clip, i)
                        # try:
                        
                            colors = DominantColors(cropped_img, clusters= 2).dominantColors()
                            # except:                                                             #Cuando solo existe un color en el recorte
                            #     colors = DominantColors(cropped_img, clusters= 1).dominantColors()
                            color_test=np.array([color.mean() for color in colors])
                            if any(color_test > 200):
                                print("Operador Detectado")                                                              #White Helmet
                            # if colors.mean() > 200:
                                # print(' OK ', [color for color in color_test if color >200],i)
                                if (detected_boxes[2]-detected_boxes[0])*(detected_boxes[3]-detected_boxes[1])>max_area:        #Max Area BBOX
                                    max_area= (detected_boxes[2]-detected_boxes[0])*(detected_boxes[3]-detected_boxes[1])
                                    max_area_index=i                            
                                # pose_results[i]['track_id']=0
                                # plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                                # plt.show()
                            # else:
                                # print(pred_clip, i)
                                # plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                                # plt.show()
                                # print(' No reconocido ',[color for color in color_test if color >220],i)
                            # else:
                            #     print("No detectado")
                        except cv2.error as e:
                            print(e)
                            pose_results=[]
                    try:
                        # print(max_area_index)
                        pose_results[max_area_index]['track_id']=0
                        del max_area_index
                    except NameError:
                        print("No se detecto nada")
                        pass

                    # print(operador)
                # except:
                #     print('Tracking error, maybe no detections here')
                #     pass
                                

                if flag_main_init ==False:

            
                    pose_results =  results= pose_results_final=[]

                    flag_main_init=True
                
            
                pose_results_last = pose_results
                next_id_last=next_id

                


                # try:
                # for result in pose_results:
                #     if result['track_id']==0:
                #         results = [result]
                #     else:
                #         results = []


                # pose_results=results
                # except:
                #     pose_results = []

                pose_id_list=[]
                for results in pose_results:
                    pose_id_list.append(results['track_id'])
                operator_index= (np.where(np.array(pose_id_list)==0))[0]

                if len(operator_index)>0:
                    pose_results_final=[pose_results[operator_index[0]]]
                    operator_counter=wait_for_operator
                    # print('SIGUE AHI ')
                if len(operator_index)==0:
                    if operator_counter<=wait_for_operator and operator_counter>0:
                        pose_results_final=pose_results_final
                        operator_counter-=1

                        # print('ESPERANDO, Contador: ', operator_counter)

                    if operator_counter==0:
                        pose_results_final=[]
                        # print('SE FUE')
            else:
                pose_results_final=pose_results  
                

                ################################################################################################3
            # pose_results=[]    
            # show pose estimation results
            self.vis_result = vis_pose_result(self.pose_model,
                                    img,
                                    pose_results_final,
                                    radius=10,
                                    thickness=4,
                                    dataset=self.pose_model.cfg.data.test.type,
                                    show=False)

            # Pose processing



                # vis_img = vis_pose_tracking_result(
                #     pose_model,
                #     img,
                #     results,
                #     # radius=4,
                #     # thickness=args.thickness,
                #     dataset=pose_model.cfg.data.test.type,
                #     # kpt_score_thr=args.kpt_thr,
                #     show=False)

            FONTFACE = cv2.FONT_HERSHEY_DUPLEX
            FONTSCALE = 2
            FONTCOLOR = (0, 0, 255)  # BGR, white
            THICKNESS = 3
            LINETYPE = 1



        
            #####################################################ACTION POSEC3D######################################################
            try:
                left_wrist=(pose_results_final[0]['keypoints'][9][:2]).astype(int)
                right_wrist=(pose_results_final[0]['keypoints'][10][:2]).astype(int)
                left_ankle=(pose_results_final[0]['keypoints'][15][:2]).astype(int)
                right_ankle=(pose_results_final[0]['keypoints'][16][:2]).astype(int)
                zonita=''
                rejilla_up=rejilla_down=suelo_rejilla=suelo_gabinete=primer_gabinete=False
                # for segmento in zona:
                if any(np.equal(left_ankle,self.zona['suelo_rejilla']).all(1)) or any(np.equal(right_ankle,self.zona['suelo_rejilla']).all(1)):
                    zonita=zonita + 'En zona vasija '
                    suelo_rejilla=True
                if any(np.equal(left_ankle,self.zona['suelo_gabinete']).all(1)) or any(np.equal(right_ankle,self.zona['suelo_gabinete']).all(1)):
                    zonita=zonita + 'En zona gabinete '
                    suelo_gabinete=True
                if any(np.equal(left_wrist,self.zona['rejilla_down']).all(1)) or any(np.equal(right_wrist,self.zona['rejilla_down']).all(1)):
                    zonita=zonita+'y toca rejilla inferior '
                    rejilla_down=True
                    # print(zonita)
                if any(np.equal(left_wrist,self.zona['rejilla_up']).all(1)) or any(np.equal(right_wrist,self.zona['rejilla_up']).all(1)):
                    zonita=zonita+ 'y toca rejilla superior '
                    rejilla_up=True
                    # print(zonita)
                if any(np.equal(left_wrist,self.zona['primer_gabinete']).all(1)) or any(np.equal(right_wrist,self.zona['primer_gabinete']).all(1)):
                    zonita=zonita+ 'y toca gabinete '
                    primer_gabinete=True

                    # print(zonita)
                if not zonita=='':
                    # print(zonita)
                    cv2.putText(self.vis_result,zonita, (50, 1000
                    ), FONTFACE, 1.5,
                        FONTCOLOR, THICKNESS, LINETYPE)

                ##Rejilla
                if rejilla_up and suelo_rejilla :
                    print('PARTIDA UP')
                    if Partida_r_up and counter<(wait_for_action-4):
                        counter=wait_for_action
                        pose_frames=[]
                        del fake_anno
                    Partida_r_up=True
                if rejilla_down and suelo_rejilla:
                    print('PARTIDA DOWN')
                    if Partida_r_down and counter<(wait_for_action-4):
                        counter=wait_for_action
                        pose_frames=[]
                        del fake_anno
                    Partida_r_down=True
                
                #Gabinete y Botonera
                if primer_gabinete and suelo_gabinete:
                    print('PRIMER GABINETE')
                    if not Partida_zona_g_1:
                        counter=wait_for_action
                        pose_frames=[]
                        vid_frames=[]
                    Partida_zona_g_1=True

                if Partida_zona_g_1 and counter<wait_for_action-9:
                    print('Gab Verificado')
                    if not suelo_gabinete:
                        Partida_zona_g_1=False
                        counter=wait_for_action
                        pose_frames=[]
                        vid_frames=[]
                        del fake_anno
                    if suelo_gabinete:
                        Partida_g_1=True

                #Comienzo de segmento
                if Partida_r_down or Partida_r_up or Partida_zona_g_1:    
                    print('COUNTER: ', counter)
                    if (counter==0 and Partida_r_up==True and Partida_r_down==False) or (counter==0 and Partida_r_down==True and Partida_r_up==False): 
                        Partida_r_up=False
                        Partida_r_down=False
                        counter=wait_for_action
                        pose_frames=[]
                        vid_frames=[]
                        del fake_anno
                ######################### Generando fake_anno ###############################
                    window=wait_for_action-counter+1
                    pose_frames.append(pose_results_final)
                    num_frame=window


                    fake_anno = dict(
                        frame_dir='',
                        label=-1,
                        img_shape=(img.shape[0],img.shape[1]),
                        original_shape=(img.shape[0],img.shape[1]),
                        start_index=0,

                        modality='Pose',
                        total_frames=num_frame)
                    num_person = max([len(x) for x in pose_frames])
                    # Current PoseC3D models are trained on COCO-keypoints (17 keypoints)
                    num_keypoint = 17
                    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                                        dtype=np.float16)
                    keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                                            dtype=np.float16)
                    
                    for i, poses in enumerate(pose_frames):
                        for j, pose in enumerate(poses):
                            pose = pose['keypoints']
                            keypoint[j, i] = pose[:, :2]
                            keypoint_score[j, i] = pose[:, 2]
                    fake_anno['keypoint'] = keypoint
                    fake_anno['keypoint_score'] = keypoint_score

                    counter-=1
                    
                ####################################################################
                    if (Partida_r_down and Partida_r_up) or Partida_g_1:
                        print('ES UNA ACCION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        # torch.cuda.empty_cache()
                        Partida_g_1=Partida_zona_g_1= Partida_r_up= Partida_r_down=False
                        counter=wait_for_action
                        print('pose_frames:   ', len(pose_frames))
                        pose_frames=[]

                        cv2.putText(self.vis_result, ('ANALIZANDO ACCION....'), (20, 50), FONTFACE, 1,
                        (0,255,0), THICKNESS, LINETYPE)

                        
                        queue_anno.put(fake_anno)
                        del fake_anno
                        
                        
                
            except:
                pass
            if not queue_action.empty():
                action=queue_action.get()
                actual_action={'name':action_name[action[0][0]], 'score':(action[0][1])*100}
                print('Se recibio acción: {}, {}%'.format(actual_action['name'], actual_action['score']))
                cv2.putText(self.vis_result, (" {0} : {1:.1f}%".format(actual_action['name'],actual_action['score'])), (200, 400), FONTFACE, 2.5,
                FONTCOLOR, 8, LINETYPE)
                
            ## Logica de riesgo

                if action[0][0]==0 and result_actions[0]==1:    
                    result_actions[0]=0
                if action[0][0]==2:
                    result_actions[0]=1
                if action[0][0]==3 and result_actions[0]==1:
                    risk='Presion en membranas'
                    btaudio.play('Riesgo de alta presión, porfavor cerrar la rejilla')

                if action[0][0]==4:
                    risk='Electrificación'
                    btaudio.play('Riesgo de electrificación, porfavor cerrar gabinete')
                  
            
            ################################################Data Frame###########################################################
            if len(pose_results_final)>0:
                op_present='Si'
            else:
                op_present='No'


            if old_op_present!=op_present or actual_action['name']!='No':
                
                self.df=self.df.append(pd.DataFrame({'Op. Presente':[op_present], 'Accion':[actual_action['name']], 'Riesgo': [risk],'Hora':[datetime.today().time().isoformat('seconds')], 'Fecha':[datetime.today().date()]}))
                self.WriteFrame(tableview, self.df)


            old_op_present=op_present
            old_action=actual_action['name']=risk='No'
            


            

            
            
            



            ########################################################################################################################3

            cv2image = cv2.cvtColor(cv2.resize(self.vis_result, (int(labelVideo.winfo_width()), int(labelVideo.winfo_height()))), cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            labelVideo.imgtk = imgtk
            labelVideo.configure(image=imgtk)

            showActions.update()
            showActions.update_idletasks()
                                        

            # # reduce image size
            # # vis_result = cv2.resize(vis_result, dsize=None, fx=0.7, fy=0.7)

            # finish=time.perf_counter() 
            # # print('Time: ', (finish-start))
            

        
        self.cam.stop()


