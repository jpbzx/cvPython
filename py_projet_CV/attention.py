import time

class AttentionTracker:
    def __init__(self, timeout=2):
        self.last_seen = time.time()
        self.timeout = timeout

    def update(self, face_detected):
        if face_detected:
            self.last_seen = time.time()
            return "ATTENTIVE"
        else:
            if time.time() - self.last_seen > self.timeout:
                return "NOT ATTENTIVE"
            return "ATTENTIVE"
