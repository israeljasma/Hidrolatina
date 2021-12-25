import pyttsx3
import torch.multiprocessing as mp

class BTAudio():
    def __init__(self):
            # print('proceso')
            self.queue_audio=mp.Queue()

    def playAudio(self, queue_audio):
        """VOICE"""
        engine= pyttsx3.init()

        """ RATE"""
        rate = engine.getProperty('rate')   # getting details of current speaking rate
        print (rate)                        #printing current voice rate
        engine.setProperty('rate', 190)    # setting up new voice rate
        voices = engine.getProperty('voices')       #getting details of current voice
        engine.setProperty('voice', voices[0].id)   
        while True:
            if not self.queue_audio.empty():
                audioOut=queue_audio.get()
                engine.say(audioOut)
                engine.runAndWait()
                engine.stop()

    def play(self, text):
        self.queue_audio.put(text)