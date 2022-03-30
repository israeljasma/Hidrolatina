import pyttsx3
import torch.multiprocessing as mp
from threading import Thread

import pyaudio
import wave
import audioop
from collections import deque
import os
import time
import math
import speech_recognition as sr

import socket
import select

import pywhatkit



class BTAudio():
    def __init__(self):
            # print('proceso')
            self.queue_audio_out=mp.Queue()
            self.queue_audio_in=mp.Queue()
            self.flag_instructivo=False
    
    def Load(self):
        self.thread_out= Thread(target=self.playAudio, args=())
        self.thread_out.start()
        self.thread_server= Thread(target=self.writeServer, args=())
        self.thread_server.start()


        self.thread_in= Thread(target=self.listenAudio, args=())
        self.thread_in.start()
     

    def playAudio(self):
        def onWord(name, location, length):
            if not self.queue_audio_out.empty():
                self.flagPlay=self.queue_audio_out.get()
                if self.flagPlay==0:
                    print('Reproducción Interrumpida')
                    engine.stop()
                else:
                    self.queue_audio_out.put(self.flagPlay)
        while True:
            if not self.queue_audio_out.empty():
                audioOut=self.queue_audio_out.get()
                if type(audioOut)==str:
                    try:
                        del engine
                    except NameError:
                        pass
                    engine= pyttsx3.init()
                    engine.setProperty('rate', 190)    # setting up new voice rate
                    voices = engine.getProperty('voices')       #getting details of current voice
                    engine.setProperty('voice', voices[0].id)
                    engine.connect('started-word', onWord)
                    engine.say(audioOut)
                    engine.runAndWait()
                    engine.stop()

                if  audioOut==0:

                    self.flagPlay=0  
                else:
                    # print('Porfavor ingrese String a reproducir, ingresaste: ',audioOut )
                    pass

    def play(self, text):
        self.queue_audio_out.put(text)
    def stop_play(self):
        self.queue_audio_out.put(0)

    def initServer(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 8000
        CHUNK = 1024

        HOST_IP = [(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]
        PORT=4444

        audio = pyaudio.PyAudio()

        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.bind((HOST_IP,PORT))
        self.serversocket.listen(5)


        def callback(in_data, frame_count, time_info, status):
            for s in self.read_list[1:]:
                try:
                    s.send(in_data)
                except:
                    pass
            return (None, pyaudio.paContinue)


        # start Recording
        self.stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, stream_callback=callback)

        # stream.start_stream()

        self.read_list = [self.serversocket]
        print (f"Server: Listening on {HOST_IP}:{PORT}...")

    def writeServer(self):
        self.initServer()
        while True:
            readable, writable, errored = select.select(self.read_list, [], [])
            for s in readable:
                if s is self.serversocket:
                    (clientsocket, address) = self.serversocket.accept()
                    self.read_list.append(clientsocket)
                    print ("Server: Connection from", address)
                else:
                    try:
                        data = s.recv(1024)
                        if not data:
                            print('Server: Se desconecto cliente', address)
                            self.read_list.remove(s)
                    except:
                        print('Se desconecto cliente', address)
                        s.close()
                        self.read_list.remove(s)
                        pass


    def initClient(self):
        print('Client: Connecting ...')
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect(('192.168.1.107', 4444))
                print('Client: Connected to Socket')
                break

            except (ConnectionRefusedError, TimeoutError):
                # self.listen_bool=False
                print('Client: Error de conexión, reconectando ...')
                time.sleep(5)
                pass
        return s

    def readClient(self):
        try:
            data = self.s.recv(self.CHUNK)
            return data
        except ConnectionResetError:
            print('Cliente: Conexión perdida')
            self.s.close()
            self.s=self.initClient()
        

    def listenAudio(self):
        self.listen_bool=False
        self.CHUNK = 1024  


        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.THRESHOLD = 550
        self.SILENCE_LIMIT = 2.5

        self.PREV_AUDIO = 1

        self.s=self.initClient()



        def test_ambient_noise(num_samples=50):
            """ Gets average audio intensity of your mic sound. You can use it to get
                average intensities while you're talking and/or silent. The average
                is the avg of the 20% largest intensities recorded.
            """
            per_max_inten=1.0   #largest intensities recorded
            print ("Getting intensity values from mic.")
            p = pyaudio.PyAudio()

            stream = p.open(format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            frames_per_buffer=self.CHUNK)

            values = [math.sqrt(abs(audioop.avg(stream.read(self.CHUNK), 4))) 
                    for x in range(num_samples)] 
            values = sorted(values, reverse=True)
            r = sum(values[:int(num_samples * per_max_inten)]) / int(num_samples * per_max_inten)
            print (" Finished ")
            print (" Average audio intensity is ", r)
            stream.close()
            p.terminate()
            return r


        def listen_for_speech(threshold=self.THRESHOLD, num_phrases=1):
            """
            Listens to Microphone, extracts phrases from it and sends it to 
            Google's TTS service and returns response. a "phrase" is sound 
            surrounded by silence (according to threshold). num_phrases controls
            how many phrases to process before finishing the listening process 
            (-1 for infinite). 
            """

            # #Open stream
            # p = pyaudio.PyAudio()

            # stream = p.open(format=self.FORMAT,
            #                 channels=self.CHANNELS,
            #                 rate=self.RATE,
            #                 input=True,
            #                 frames_per_buffer=self.CHUNK,
            #                 input_device_index=1)

            print ("* Listening mic. ")
            audio2send = []
            cur_data = ''  # current chunk  of audio data
            rel = self.RATE/self.CHUNK
            slid_win = deque(maxlen=int(self.SILENCE_LIMIT * rel))
            #Prepend audio from 0.5 seconds before noise was detected
            prev_audio = deque(maxlen=int(self.PREV_AUDIO * rel)) 
            started = False
            n = num_phrases
            response = []

            while (num_phrases == -1 or n > 0):
                # cur_data = stream.read(self.CHUNK)
                cur_data= self.readClient()
                # print(cur_data)
                try:
                    slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))
                except TypeError:
                    listen_for_speech()
                if not self.queue_audio_in.empty():
                    self.listen_bool=self.queue_audio_in.get()
                #print slid_win[-1]
                if(sum([x > self.THRESHOLD for x in slid_win]) > 0):
                    if(not started):
                        #print "Starting record of phrase"
                        started = True
                    audio2send.append(cur_data)
                elif (started is True):
                    #print "Finished"
                    # The limit was reached, finish capture and deliver.
                    try:
                        filename = save_speech(list(prev_audio) + audio2send)
                    except TypeError:
                        listen_for_speech()
                    # Send file to Google and get response
                    r = recognition_speech(filename) 
                    if num_phrases == -1:
                        print ("Response", r)
                    else:
                        response.append(r)
                    # Do some stuff with recognition
                    action_speech(response)
                    # Remove temp file. Comment line to review.
                    os.remove(filename)
                    # Reset all
                    started = False
                    slid_win = deque(maxlen=int(self.SILENCE_LIMIT * rel))
                    prev_audio = deque(maxlen=int(0.5 * rel)) 
                    audio2send = []
                    n -= 1
                    #print "Listening ..."
                elif (self.listen_bool==False):
                    print('Forced Close')
                    break
                else:
                    prev_audio.append(cur_data)

            print ("* Done recording")

            return response

        def recognition_speech(file):
            r = sr.Recognizer() 
        
            with sr.AudioFile(file) as source:
                audio = r.record(source)  # read the entire audio file

                try:
                    text = r.recognize_google(audio, language="es-CL", show_all=True )
                    print('You said: {}'.format(text))
                except:
                    print('Sorry could not hear')
                    text=None
            return text
        def action_speech(speech):
            commands=''
            
            try:
                for alt in enumerate(speech[0]['alternative']):
                    # print('DIJISTE: ', alt[1]['transcript'])
                    if 'Sony ' in alt[1]['transcript']:
                        commands=alt[1]['transcript']
                    if self.flag_instructivo is True:
                        if '1' in alt[1]['transcript']:
                            self.stop_play()
                            self.play('Los pasos de verificación son los siguientes   \n')
                            self.play('Area de trabajo limpia y despejada\n ponte tus epepé\n protecciones de seguridad en buen estado\n ten tus herramientas necesarias\n Verifica parada de emergencias accionada.\n Las correas de las bombas de alta presión tensadas y en correcto estado.\nEstado del aceite de las bombas P-01 y P-03.\n Bomba de recirculación P-05 conectada a la energía.\n Asegura que el sistema de líquido refrigerante esté funcionando.\n Asegura que los flexibles de alimentación a los grupos de bombeo se encuentren conectados a TK-02.\n Manten las válvulas de descarga VM-01, VM-02, junto con válvula de desagüe VM-28 de estanque TK-01 cerradas y las válvulas de descarga VM-02 y VM-04 de TK-02 abiertas.\nEstanques TK-01 y TK-02 deben tener por lo menos ¾ de agua de alimentación. De lo contrario abrir válvula de alimentación de agua hasta llegar al nivel requerido.\nVerifica que las válvulas VM-05 y VM-07 del sistema de certificación de membranas estén abiertas, y las válvulas VM-06 y VM-08 del sistema de certificación de vasijas cerradas.\n Selecciona la o las membranas para medir')
                            self.flag_instructivo=False 
                        if '2' in alt[1]['transcript']:
                            self.stop_play()
                            self.play('Los pasos de acondicionamiento son los siguientes')
                            self.play('Energizar sistema de enfriamiento levantando el switch del tablero de sector planta Riles.\n Encender la bomba de recirculación P-05 del sistema de enfriamiento, asegurándose que las válvulas VM-18 y VM-21 se encuentren abiertas, y VM-17 y VM-22 cerradas.\n Verificar que set point de temperatura del sistema de enfriamiento (chiller) esté en 20 [°C].\n Verificar que la presión antes del filtro F-03 no sobrepase 2 [bar] (PI-30), ya que de lo contrario significa que este está muy sucio y es necesario apagar la máquina y limpiar el filtro.\n Verificar que la presión antes del filtro F-04 no sobrepase 1,7 [bar] (PI-31), ya que de lo contrario significa que este está muy sucio y es necesario apagar la máquina y limpiar el filtro.\nVerificar que la presión antes del filtro F-04 no sobrepase 1 coma 7 bar (PI-31), ya que de lo contrario significa que este está muy sucio y es necesario apagar la máquina y limpiar el filtro.\nVerificar que tanto el sensor manual como en línea de conductividad presenten una conductividad entre 53,3 y 53,7 [mS/cm], siendo lo ideal 53,5 [mS/cm]. Si las mediciones no coinciden, se deben calibrar inmediatamente.\nVerificar en registro de calibración “Registro de Calibraciones LAB-F02” que sensor de pH esté calibrado, de lo contrario, se debe calibrar inmediatamente.\nVerificar que pH de agua de alimentación esté entre 7,0 y 7,7; siendo el ideal 7,3. Si pH se encuentra bajo, agregar con gotario soda. En caso de que pH se encuentre alto, agregar con gotario ácido clorhídrico.\n Antes de cargar las membranas, chequear estado de o-rings y lubricar.\n Anotar código de las membranas a medir en la “Ficha de Registro de certificación de vasijas”(PRO-FI11).\n Cargar las membranas que se van a medir, en las vasijas correspondientes (VE-01, VE-02).\n Cerrar las vasijas con tapa y anillo de seguridad.\n Verificar que cada vasija se encuentre correctamente cerrada tirando hacia atrás cada una de las tapas.\n Cerrar rejilla de protección de máquina de certificación.\nMantener abiertas válvulas de alimentación, producto y rechazo (VM-09, VM-11 y VM-13) de la vasija que contiene la membrana a medir (VE-01).\n Mantener cerradas válvulas de alimentación, producto y rechazo (VM-10, VM-12 y VM-14) de la segunda vasija (VE-02).')
                            self.flag_instructivo=False    
                        if '3' in alt[1]['transcript']:
                            self.stop_play()
                            self.play('Los pasos de operación son los siguientes')
                            self.play('Levantar protecciones de la máquina.\n Liberar parada de emergencia.\n Si la temperatura del sistema se encuentra a las condiciones requeridas a 20 grados celsius, energizar las unidades de alta presión levantando el switch del tablero eléctrico principal.\n Anotar el código de la membrana que se medirá en el panel del sistema.\n Antes de comenzar la presurización del sistema, pulsar el botón de guardado de datos en el panel (pantalla táctil), para dar comienzo al proceso de monitoreo.\n Encender bombas Booster de ambas máquinas (P-01 y P-03).\n Encender bombas de alta presión de ambas máquinas (P-02 y P-04).\n Levantar presión muy lentamente, cerrando válvula de descarga de rechazo VM-15. Este proceso debiera tomar a lo menos 1 minuto.\n Regular presión con válvula reguladora VR-02 hasta alcanzar 62 [bar], siendo aceptable valores entre 61 coma 8 y 62 coma 2 bar.\n Regular flujo de alimentación de 30 galones por minuto con válvula de recirculación VR-01.\n Verificar que el sistema no presente filtraciones.\n Operar la máquina durante 10 minutos, para que se estabilicen los parámetros. Una vez logrado el estado estacionario del sistema, pulsar el botón de guardado de datos en el panel (pantalla táctil). Luego, registrar en ficha “Registro de Mediciones Certificación Membranas Osmosis Inversa PRO-F11” conductividad, pH y temperatura de alimentación, conductividad de producto, flujo de producto y flujo de rechazo, junto a los demás parámetros indicados en la ficha PRO-F11.\n Una vez realizada la medición, bajar presión de operación con válvula reguladora VR-02, despresurizar muy lentamente el sistema con válvula de descarga del sistema VM-15 y apagar unidades de alta presión. Este proceso debiera tomar al menos 1 minuto.\n En caso de requerir realizar una segunda medición, cerrar válvulas VM-09, VM-11 y VM-13 de la primera vasija VE-01 y abrir válvulas VM-10, VM-12 y VM-14 de la segunda vasija VE-02, y repetir el procedimiento de anotar codigo de membrana.\n Si no hay otra membrana por certificar, o si se requiere cambiar de membranas, se procede a vaciar el sistema.')
                            self.flag_instructivo=False    
                        if '4' in alt[1]['transcript']:
                            self.stop_play()
                            self.play('Los pasos de vaciado son los siguientes')
                            self.play('Pulsar parada de emergencia del sistema.\n Asegurar que válvulas VM-09 y VM-10 se encuentren cerradas.\n Asegurar que las válvulas VM-11, VM-12, VM-13 y VM-14 estén abiertas.\n Verificar que VE-01 y VE-02 se encuentran despresurizados, comprobando que PI-15 esté por debajo de 1 bar.\n Abrir rejilla de protección, retirar anillos de seguridad y tapas.\n Extraer e inclinar las membranas certificadas para botar la mayor cantidad de agua.\n Desenergizar equipo bajando el switch del tablero eléctrico principal y el switch del sistema de enfriamiento en tablero planta Riles.\n Apagar chillers y consola de medición de la máquina de alta presión.')
                            self.flag_instructivo=False    
                        print('INSTRUCTIVO ACTIVADO')

                        # print(commands)
                # print(commands)
                commands= commands.lower()
                if 'sony' in commands:
                    ## Comandos
                    if 'asistencia' in commands:
                        print('PIDE ASISTENCIA')
                        pywhatkit.send_mail("equipo.vandving@gmail.com", "Hidrolatina123", "Asistencia", "Asisteme Porfavor", "javier.esuazo.s@gmail.com")
                        pywhatkit.sendwhatmsg_instantly("+56994213132", "Asistencia",5, True, 2)

                    if 'ayuda' in commands:

                        print('PIDE AYUDA')
                        self.play('pidiendo ayuda')
                        pywhatkit.send_mail("equipo.vandving@gmail.com", "Hidrolatina123", "Ayuda", "Una ayudita", "javier.esuazo.s@gmail.com")
                        pywhatkit.sendwhatmsg_instantly("+56994213132", "Ayuda",5, True, 2)
                    if 'instructivo' in commands:
                        print('SOLICITANDO INSTRUCTIVO')
                        self.flag_instructivo=True
                        self.play('¿Que numero deseas escuchar?\n uno\n  verificación\n dos\n acondicionamiento\n tres\n operación \n cuatro \n vaciado ')
                    if 'para' in commands:
                        print('Parando')
                        self.stop_play()

            except TypeError:
                print('No recognition')

        def save_speech(data):
            """ Saves mic data to temporary WAV file. Returns filename of saved 
                file """

            filename = 'output_'+str(int(time.time()))
            # writes data to WAV file
            data = b''.join(data)
            wf = wave.open(filename + '.wav', 'wb')
            wf.setnchannels(1)
            # wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setsampwidth(2)
            wf.setframerate(self.RATE)  # TODO make this value a function parameter?
            wf.writeframes(data)
            wf.close()

            # wf1 = wave.open('C:/Users/darkb/Desktop/demo2.wav', 'wb')
            # wf1.setnchannels(1)
            # wf1.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            # wf1.setframerate(self.RATE)
            # wf1.writeframes(data)
            # wf1.close()
            return filename + '.wav'

        while True:
            if not self.queue_audio_in.empty():
                self.listen_bool=self.queue_audio_in.get()
            if self.listen_bool==True:
                data = listen_for_speech()  # listen to mic.
                if type(data)==str:
                    self.queue_audio_in=data
            else:
                pass

    def listen(self):
        self.queue_audio_in.put(True)  
    def stop_listen(self):
        self.queue_audio_in.put(False)           
    
    def exit(self):
        self.s.close()

if __name__ == '__main__':


    audio=BTAudio()
    # queue_audio_out=mp.Queue()
    p1 = mp.Process(target=audio.Load, args=())
    p1.start()
    audio.listen()
    time.sleep(5)
    audio.play('primera Este es un test de audio que nose que dice pero deberia durar mas de dos segundos, o sino nose que onda')
    # print("diciendo texto")
    time.sleep(3)
    audio.stop_play()
    # print('debio parar')
    # time.sleep(3)
    audio.play('segunda Este no es un test de audio que nose que dice pero deberia durar mas de dos segundos, o sino nose que onda')
    time.sleep(3)
    audio.stop_play()

    audio.play('tercera Este no es un test de audio que nose que dice pero deberia durar mas de dos segundos, o sino nose que onda')
    time.sleep(3)
    audio.stop_play()
    while True:
        try:
            pass
        except KeyboardInterrupt:
            print('Exit Audio')
            p1.terminate()
            break
    # audio.play(0)
    
    
    # time.sleep(5)
    # audio.stop_listen()
    # time.sleep(5)
    # audio.listen()
    # time.sleep(5)
    # p1.terminate()