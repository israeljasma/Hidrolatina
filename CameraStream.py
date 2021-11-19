import time
from threading import Thread, Lock

import cv2

class CameraStream(object):
    def _init_(self, src=0, delay=0):
        self.stream = cv2.VideoCapture(src)
        self.delay=delay
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
            time.sleep(self.delay)                          #Simulation Video delay
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()    

    def read(self):
        self.read_lock.acquire()
        # frame = self.frame.copy()
        self.read_lock.release()
        return self.frame

    def stop(self):
        self.started = False
        self.stream.release()
    

    def _exit_(self, exc_type, exc_value, traceback):
        self.thread.join()