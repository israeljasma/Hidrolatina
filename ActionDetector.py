import cv2
import numpy as np
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,inference_bottom_up_pose_model,
                         vis_pose_result, process_mmdet_results, vis_3d_pose_result, inference_pose_lifter_model, extract_pose_sequence, get_track_id,vis_pose_tracking_result)

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

        self.pose_config = 'C:/hidrolatina/data/hrnet_w32_coco_384x288_udp.py'
        self.pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w32_coco_384x288_udp-e97c1a0f_20210223.pth'

        self.posec3d_config='C:/hidrolatina/data/3hl_test.py'
        self.posec3d_checkpoint='https://github.com/EquipoVandV/mmactionVandV/raw/main/TRAIN/work_dirs/posec3d/3HL/epoch_40.pth'

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
        x_cabinet=np.array(list(range(960,1040)))
        x_grating1=np.array(list(range(790,1040)))
        x_grating2=np.array(list(range(807,1040)))
        x_ground= np.array(list(range(900,1300)))
        y_ground=np.array(list(range(700,1080)))
        y_ground=np.flip(y_ground)
        #Generate y coordinates

        ramp_cabinet= lambda t: round(-0.6795*t + 964.5513)

        ramp_grating1= lambda t: round(-0.7073*t + 1087.2439)

        ramp_grating2= lambda t: round(-0.9464*t + 1620)
        ramp_ground= lambda t: round(0.3906*t + 345.0539)
        ramp_ground_y= lambda t: round((t- 1805.0445103857564)/-1.127596439169139)
        ramp_ground_2= lambda t: round(0.2500000000000047*t + 510)

        def create_zone(ramp, x,largo=10):
            y = np.array([ramp(xi) for xi in x])
            x_copy=y_copy=np.array([]).astype(int)
            for i in range(largo):
                x_copy=np.append(x_copy,x)
                y_copy=np.append(y_copy,y+i)
            return np.column_stack((x_copy, y_copy))

        def create_zone2(ramp, y,ramp2,largo=10):
            x_copy=y_copy=np.array([]).astype(int)
            for i in range(largo):
                if i==0:
                    x = np.array([ramp(xi) for xi in y])      
                    x_copy=np.append(x_copy,x+i)
                    y_copy=np.append(y_copy,y)

                if i>0:
                    xlim=x_copy[len(x_copy)-1]
                    y=np.array(list(range(ramp2(xlim+1),1080)))
                    x = np.array([ramp(xi) for xi in y])
                    x_copy=np.append(x_copy,x+i)
                    y_copy=np.append(y_copy,y)
            return np.column_stack((x_copy, y_copy))

        self.zona={ 'rejilla_up': create_zone(ramp_grating1, x_grating1, 40), 'rejilla_down': create_zone(ramp_grating2, x_grating2, 60),
                'suelo_rejilla':create_zone2(ramp_ground_y,y_ground, ramp_ground_2,500)}

    def proc_paral(self, queue_anno,queue_action, flag_posec3d_init): 

        posec3d_model=init_recognizer(self.posec3d_config, self.posec3d_checkpoint, 'cuda:0')
        # # model.to('cuda:0')
        fake_anno={'frame_dir': '', 'label': -1, 'img_shape': (10, 10), 'original_shape': (10, 10), 'start_index': 0, 'modality': 'Pose', 'total_frames': 1, 'keypoint': np.zeros((0, 1, 17, 2), dtype=np.float16), 'keypoint_score': np.zeros((0, 1, 17), dtype=np.float16)}
        actions=inference_recognizer(posec3d_model,fake_anno)
        print('Inicializó Modelo Acciones')
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

    def inferenceActionDetector(self, queue_anno, queue_action, labelVideo, showActions, cameraFrame):
        area_bbox=False
        track_flag=True
        wait_for_operator=5
        operator_counter=wait_for_operator


        next_id =next_id_last= 2

        pose_results=pose_results_last = pose_results_list= pose_frames=[]
        flag_main_init=False

        window=5
        
        wait_for_action=20
        counter=wait_for_action
        Partida_r_down=Partida_r_up=ACCION_WARN= False

        candidate_captions=['white', 'color']
        action_name=['Rejilla cerrada!','Accion Aleatoria','Rejilla abierta!']
        
        
        
        torch.cuda.empty_cache()
        self.cam = CameraStream('C:/hidrolatina/test_beta2.mp4',delay=0.03).start()    #test
        # cam = CameraStream('C:/Users/Hidrolatina/Downloads/Videos_dataset/2.mp4',delay=0.03).start()  #test
        # cam = CameraStream('C:/Users/Hidrolatina/Downloads/Videos_dataset/CAM02 acciones 60 ciclos 12-10-2021.wmv', delay=0.03).start()    #t
        # cam = CameraStream(0).start() 
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

            if area_bbox == True:
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
            # for segmento in zona:
            #     if segmento=='suelo_rejilla':
            #         # for point in zona[segmento]:
            #         #     img = cv2.circle(img, (point), radius=1, color=(255, 0, 0), thickness=-1)
            #             pass
            #     else: 
            #         for point in zona[segmento]:
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
                        rad=((ymax-ymin)/2)*0.2
                        # print('Radius: ', rad)
                        
                        # cropped_img = img[int(ymin+3*rad):int(ymax-3*rad),(int(head_point[0]-rad)):(int(head_point[0]+rad))]
                        cropped_img = img[int(ymin+3*rad):int(ymax-3*rad),(int(head_point[0]+6)):(int(head_point[0]+9))] 
                        
                        
                        print(len(cropped_img))
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
                    try:
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
                rejilla_up=rejilla_down=suelo_rejilla=False
                # for segmento in zona:
                if any(np.equal(left_ankle,self.zona['suelo_rejilla']).all(1)) or any(np.equal(right_ankle,self.zona['suelo_rejilla']).all(1)):
                    zonita=zonita + 'En zona '
                    suelo_rejilla=True
                if any(np.equal(left_wrist,self.zona['rejilla_down']).all(1)) or any(np.equal(right_wrist,self.zona['rejilla_down']).all(1)):
                    zonita=zonita+'y toca rejilla inferior '
                    rejilla_down=True
                    # print(zonita)
                if any(np.equal(left_wrist,self.zona['rejilla_up']).all(1)) or any(np.equal(right_wrist,self.zona['rejilla_up']).all(1)):
                    zonita=zonita+ 'y toca rejilla superior '
                    rejilla_up=True
                    # print(zonita)

                    # print(zonita)
                if not zonita=='':
                    # print(zonita)
                    cv2.putText(self.vis_result,zonita, (50, 1000
                    ), FONTFACE, 1.5,
                        FONTCOLOR, THICKNESS, LINETYPE)

                
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
                
                
                if Partida_r_down or Partida_r_up:    
                    print('COUNTER: ', counter)
                    if counter==0 and Partida_r_up==True and Partida_r_down==False or counter==0 and Partida_r_down==True and Partida_r_up==False: 
                        Partida_r_up=False
                        Partida_r_down=False
                        counter=wait_for_action
                        pose_frames=[]
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
                    if Partida_r_down and Partida_r_up:
                        print('ES UNA ACCION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        # torch.cuda.empty_cache()
                        Partida_r_up=False
                        Partida_r_down=False
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
                print('Se recibio acción: ', action)
                cv2.putText(self.vis_result, (" {0} : {1:.1f}%".format(action_name[action[0][0]],(action[0][1])*100)), (200, 400), FONTFACE, 2.5,
                FONTCOLOR, 8, LINETYPE)
                ACCION_WARN=False
                
                # time.sleep(4)   

            


            ########################################################################################################################3

            cv2image = cv2.cvtColor(cv2.resize(self.vis_result, (int(cameraFrame.winfo_width()), int(cameraFrame.winfo_height()))), cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            labelVideo.imgtk = imgtk
            labelVideo.configure(image=imgtk)

            showActions.update()
            showActions.update_idletasks()
            
            
            # # vis_img= cv2.resize(vis_img, (810,540))
            # # vis_result= cv2.resize(vis_result, (810,540))
            # vis_result= cv2.resize(vis_result, (900,512))
            # cv2.imshow('Live Test', vis_result)
            # cv2.setWindowProperty('Live Test', cv2.WND_PROP_TOPMOST, 1)
            

                                        

            # # reduce image size
            # # vis_result = cv2.resize(vis_result, dsize=None, fx=0.7, fy=0.7)

            # finish=time.perf_counter() 
            # # print('Time: ', (finish-start))
            
            # if cv2.waitKey(1) == 27: 
            #     break

            # if ACCION_WARN==True:
            #     time.sleep(1)
            #     ACCION_WARN=False
            #     # print('Deberia verse la accion, pero parece que no')
        
        self.cam.stop()