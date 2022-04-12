from asyncore import loop
import pandas as pd
import time
import paho.mqtt.client as mqtt
from sqlalchemy import true
import torch.multiprocessing as mp
from datetime import datetime
import time
import json
import socket
from Services import API_Services

class Sensors(object):
    def __init__(self, btAudio):
        self.btAudio=btAudio
        self.queue=mp.Queue()
        self.token=''
        
    # The callback for when the client receives a CONNACK response from the server.

    def Load(self):
        while True:
            if not self.queue.empty():
                var = self.queue.get()
                if var == 1:
                    self.client_pres = mqtt.Client()
                    self.client_pres.on_connect = self.connect_pres
                    self.client_pres.on_message = self.msg_pres
                    self.client_pres.connect("192.168.100.150", 1883, 60)
                    self.client_pres.loop_forever()

    def connect_pres(self, client, userdata, flags, rc):
        print("Connected with result code ") 
        topics=[("iot-2/type/MT8102iE/id/HMI-F7F9/evt/topic 1/fmt/json",0)]
        client.subscribe(topics)

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

        if payload['MEM_CaudalRech'][0]>100.0:
            self.btAudio.play("Peligro: valor alto de conductividad de alimentacion")
        if payload['MEM_CaudalRech'][0]<25.0:
            self.btAudio.play("Peligro: valor bajo de conductividad de alimentacion")

            
        if payload['MEM_CaudalPerm'][0]>100.0:
            self.btAudio.play('Peligro: valor alto de caudal de permeado')
        if payload['MEM_CaudalPerm'][0]<25.0:
            self.btAudio.play('Peligro: valor bajo de caudal de permeado')
            
        if payload['MEM_Cond_Perm'][0]>1000.0:
            self.btAudio.play('Peligro: valor alto de conductividad de permeado')
        if payload['MEM_Cond_Perm'][0]<100.0:
            self.btAudio.play('Peligro: valor bajo de conductividad de permeado')
   
            
        if payload['CondAlimen'][0]>56.0:
            self.btAudio.play('Peligro: valor alto de conductividad de alimentacion')
        if payload['CondAlimen'][0]<50.0:
            self.btAudio.play('Peligro: valor bajo de conductividad de alimentacion')

        if payload['MEM_PresAlimen'][0]>65.0:
            self.btAudio.play('Peligro: valor alto de presion de alimentacion')
        if payload['MEM_PresAlimen'][0]<60.0:
            self.btAudio.play('Peligro: valor bajo de presion de alimentacion')
            
        
        if payload['MEM_PresRecha'][0]>63.0:
            self.btAudio.play('Peligro: valor alto de presion de rechazo')
        if payload['MEM_PresRecha'][0]<58.0:
            self.btAudio.play('Peligro: valor bajo de presion de rechazo')
            
        if payload['TempAlimen'][0]>25.0:
            self.btAudio.play('Peligro: valor alto de temperatura de alimentacion')
        if payload['TempAlimen'][0]<10.0:
            self.btAudio.play('Peligro: valor bajo de temperatura de alimentacion')

        self.loop = True

        if not self.queue.empty():
                var = self.queue.get()
                if var == 0:                   
                    self.client_pres.disconnect()
                    self.Load()

    def startSensors(self):
        self.queue.put(1)

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

# while True:
#     try:
#         Sensors()

#     except socket.timeout:
#         print('Error de Conexión, Reintentando')
#         time.sleep(10)
#         pass
#     except KeyboardInterrupt:
#         print('---------------------------STOPPED---------------------------')
#         break