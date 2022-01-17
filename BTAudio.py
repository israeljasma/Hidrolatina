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

class BTAudio():
    def __init__(self):
            # print('proceso')
            self.queue_audio_out=mp.Queue()
            self.queue_audio_in=mp.Queue()
    
    def Load(self):
        self.thread_out= Thread(target=self.playAudio, args=())
        self.thread_out.start()
        self.thread_in= Thread(target=self.listenAudio, args=())
        self.thread_in.start()

    def playAudio(self):
        """VOICE"""
        engine= pyttsx3.init()

        """ self.RATE"""
        # rate = engine.getProperty('rate')   # getting details of current speaking rate
        # print (rate)                        #printing current voice rate
        engine.setProperty('rate', 190)    # setting up new voice rate
        voices = engine.getProperty('voices')       #getting details of current voice
        engine.setProperty('voice', voices[0].id)   
        while True:
            if not self.queue_audio_out.empty():
                audioOut=self.queue_audio_out.get()
                if audioOut==0:              #EXIT
                    break
                if type(audioOut)==str:
                    engine.say(audioOut)
                    engine.runAndWait()
                    engine.stop()
                else:
                    print('Porfavor ingrese String a reproducir')

    def play(self, text):
        self.queue_audio_out.put(text)

    def listenAudio(self):
        self.listen_bool=False
        self.CHUNK = 1024  
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 8000
        self.THRESHOLD = 4000 
        self.SILENCE_LIMIT = 1.5  
        self.PREV_AUDIO = 1


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

            #Open stream
            p = pyaudio.PyAudio()

            stream = p.open(format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            frames_per_buffer=self.CHUNK,
                            input_device_index=1)

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
                cur_data = stream.read(self.CHUNK)
                slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))
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
                    filename = save_speech(list(prev_audio) + audio2send, p)
                    # Send file to Google and get response
                    r = recognition_speech(filename) 
                    if num_phrases == -1:
                        print ("Response", r)
                    else:
                        response.append(r)
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
            stream.close()
            p.terminate()

            return response

        def recognition_speech(file):
            r = sr.Recognizer() 
        
            with sr.AudioFile(file) as source:
                audio = r.record(source)  # read the entire audio file

                try:
                    text = r.recognize_google(audio, language="es-CL", show_all=False )
                    print('You said: {}'.format(text))
                except:
                    print('Sorry could not hear')
                    text=None
            return text

        def save_speech(data, p):
            """ Saves mic data to temporary WAV file. Returns filename of saved 
                file """

            filename = 'output_'+str(int(time.time()))
            # writes data to WAV file
            data = b''.join(data)
            wf = wave.open(filename + '.wav', 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
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
    
