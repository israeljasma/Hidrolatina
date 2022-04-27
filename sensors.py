import time
import paho.mqtt.client as mqtt
import torch.multiprocessing as mp
import time
import json
import socket
from Services import API_Services

class Sensors(object):
    def __init__(self, btAudio):
        self.btAudio=btAudio
        self.queue=mp.Queue()
        self.queue_token=mp.Queue()

        self.token=''
        
    # The callback for when the client receives a CONNACK response from the server.

    def Load(self):
        while True:
            if not self.queue.empty():
                var = self.queue.get()
                if var == 1:
                    if not self.queue_token.empty():
                        self.token=self.queue_token.get()
                    self.client_pres = mqtt.Client()
                    self.client_pres.on_connect = self.connect_pres
                    self.client_pres.on_message = self.msg_pres
                    self.client_pres.connect("192.168.100.150", 1883, 60)
                    self.client_pres.loop_forever()

    def connect_pres(self, client, userdata, flags, rc):
        print("Connected with result code ") 
        topics=[("iot-2/type/MT8102iE/id/HMI-F7F9/evt/topic 1/fmt/json",0)]
        client.subscribe(topics)
        self.flag_MEM_CaudalRech=self.flag_MEM_CaudalPerm=self.flag_MEM_Cond_Perm=self.flag_CondAlimen=self.flag_MEM_PresAlimen=self.flag_MEM_PresRecha=self.flag_TempAlimen=self.flag_estacionario=False
        self.counter=0
        self.reg_mem=False

    def msg_pres(self, client, userdata, msg):
        payload= json.loads(msg.payload.decode("utf-8"))
        # value=payload['MEM_PresAlimen'][0]
        # print(payload)
        # time.sleep(1)  #Tiempo de muestreo

        API_Services.membraneRejectionFlow(payload['MEM_CaudalRech'][0], payload['ts'], self.token)
        API_Services.membranePermeate(payload['MEM_CaudalPerm'][0], payload['ts'], self.token)
        API_Services.conductivityPermeateMembranes(payload['MEM_Cond_Perm'][0], payload['ts'], self.token)
        API_Services.feedConductivity(payload['CondAlimen'][0], payload['ts'], self.token)
        API_Services.membraneFeedPressure(payload['MEM_PresAlimen'][0], payload['ts'], self.token)
        API_Services.membraneRejectionPressure(payload['MEM_PresRecha'][0], payload['ts'], self.token)
        API_Services.feedTemperature(payload['TempAlimen'][0], payload['ts'], self.token)
        API_Services.conductivityPermeateMembranes(payload['MEM_Cond_Perm'][0], payload['ts'], self.token)
        API_Services.registerMembranes(payload['GuardaMem'][0], payload['ts'], self.token)
        # print('FLAG CAUDAL DE RECHAZO: ', self.flag_MEM_CaudalRech)

        if not self.flag_estacionario:
            if payload['MEM_PresAlimen'][0]>=60.0:
                self.counter+=1
                if self.counter>=3:
                    self.flag_estacionario=True
                    self.counter=0

            self.now_press_alim = payload['MEM_PresAlimen'][0]
            try: 
                self.last_press_alim
                sum = self.now_press_alim - self.last_press_alim
                vel_press = sum/2
                if vel_press > 1:
                    self.btAudio.play('Peligro: Presurizar mas lentamente a solo 1 bar por segundo')
                if vel_press < -1:
                    self.btAudio.play('Peligro: Despresurizar mas lentamente a solo 1 bar por segundo')
            except AttributeError:
                self.last_press_alim = self.now_press_alim
            




        if self.flag_estacionario:
            if payload['GuardaMem'][0]==True:
                self.reg_mem=True

            if payload['MEM_CaudalRech'][0]>100.0 and not self.flag_MEM_CaudalRech:
                self.btAudio.play("Peligro: valor alto de conductividad de alimentacion")
                self.flag_MEM_CaudalRech=True
            if payload['MEM_CaudalRech'][0]<25.0 and not self.flag_MEM_CaudalRech:
                self.btAudio.play("Peligro: valor bajo de conductividad de alimentacion")
                self.flag_MEM_CaudalRech=True
            elif 25.0<=payload['MEM_CaudalRech'][0]<=100.0:
                self.flag_MEM_CaudalRech=False
                
            if payload['MEM_CaudalPerm'][0]>100.0 and not self.flag_MEM_CaudalPerm:
                self.btAudio.play('Peligro: valor alto de caudal de permeado')
                self.flag_MEM_CaudalPerm=True
            if payload['MEM_CaudalPerm'][0]<25.0 and not self.flag_MEM_CaudalPerm:
                self.btAudio.play('Peligro: valor bajo de caudal de permeado')
                self.flag_MEM_CaudalPerm=True
            elif 25.0<=payload['MEM_CaudalPerm'][0]<=100.0:
                self.flag_MEM_CaudalPerm=False
                
            if payload['MEM_Cond_Perm'][0]>1000.0 and not self.flag_MEM_Cond_Perm:
                self.btAudio.play('Peligro: valor alto de conductividad de permeado')
                self.flag_MEM_Cond_Perm=True
            if payload['MEM_Cond_Perm'][0]<100.0 and not self.flag_MEM_Cond_Perm:
                self.btAudio.play('Peligro: valor bajo de conductividad de permeado')
                self.flag_MEM_Cond_Perm=True
            elif 100.0<=payload['MEM_Cond_Perm'][0]<=1000.0:
                self.flag_MEM_Cond_Perm=False
                
            if payload['CondAlimen'][0]>56.0 and not self.flag_CondAlimen:
                self.btAudio.play('Peligro: valor alto de conductividad de alimentacion')
                self.flag_CondAlimen=True
            if payload['CondAlimen'][0]<50.0 and not self.flag_CondAlimen:
                self.btAudio.play('Peligro: valor bajo de conductividad de alimentacion')
                self.flag_CondAlimen=True
            elif 50.0<=payload['CondAlimen'][0]<=56.0:
                self.flag_CondAlimen=False

            if payload['MEM_PresAlimen'][0]>65.0 and not self.flag_MEM_PresAlimen:
                self.btAudio.play('Peligro: valor alto de presion de alimentacion')
                self.flag_MEM_PresAlimen=True
            if payload['MEM_PresAlimen'][0]<60.0 and not self.flag_MEM_PresAlimen:
                self.btAudio.play('Peligro: valor bajo de presion de alimentacion')
                self.flag_MEM_PresAlimen=True

            elif 60.0<=payload['MEM_PresAlimen'][0]<=65.0:
                self.flag_MEM_PresAlimen=False
                
            
            if payload['MEM_PresRecha'][0]>63.0 and not self.flag_MEM_PresRecha:
                self.btAudio.play('Peligro: valor alto de presion de rechazo')
                self.flag_MEM_PresRecha=True
            if payload['MEM_PresRecha'][0]<58.0 and not self.flag_MEM_PresRecha:
                self.btAudio.play('Peligro: valor bajo de presion de rechazo')
                self.flag_MEM_PresRecha=True
            elif 58.0<=payload['MEM_PresRecha'][0]<=63.0:
                self.flag_MEM_PresRecha=False
                
            if payload['TempAlimen'][0]>25.0 and not self.flag_TempAlimen:
                self.btAudio.play('Peligro: valor alto de temperatura de alimentacion')
                self.flag_TempAlimen=True
            if payload['TempAlimen'][0]<10.0 and not self.flag_TempAlimen:
                self.btAudio.play('Peligro: valor bajo de temperatura de alimentacion')
                self.flag_TempAlimen=True
            elif 10.0<=payload['TempAlimen'][0]<=25.0:
                self.flag_TempAlimen=False

            if payload['MEM_PresAlimen'][0]<60.0 and self.reg_mem:
                self.flag_estacionario=False
                self.reg_mem=False


        if not self.queue.empty():
                var = self.queue.get()
                if var == 0:                   
                    self.client_pres.disconnect()
                    self.Load()

    def startSensors(self):
        self.queue.put(1)
    
    def sendToken(self, token):
        self.queue_token.put(token)

    def stopSensors(self):
        self.queue.put(0)


if __name__== '__main__':
    from BTAudio_DuplexSockets import BTAudio    
    btaudio=BTAudio()
    sensors=Sensors(btaudio)      
    p1 = mp.Process(target=btaudio.Load, args=())
    p1.start()

    p0 = mp.Process(target=sensors.Load, args=())
    p0.start()

    while True:
        try:
            Sensors(btaudio)

        except socket.timeout:
            print('Error de Conexión, Reintentando')
            time.sleep(10)
            pass
        except KeyboardInterrupt:
            print('---------------------------STOPPED---------------------------')
            break