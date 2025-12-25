import time
import alsaaudio

mixer = alsaaudio.Mixer()
normal_volume = mixer.getvolume()[0]

class SmileTracker:
    def __init__(self, timeout = 0.5):
        self.last_seen = time.time()
        self.timeout = timeout

    def update(self, hand_detected):
        if hand_detected:
            self.last_seen = time.time()
            mixer.setvolume(20)
            return "20 %"
        
        else:
            if time.time() - self.last_seen > self.timeout:
                mixer.setvolume(normal_volume)

                return "Normal"
            return "20 %"