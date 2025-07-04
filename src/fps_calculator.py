import time

class FPSCalculator:
    def __init__(self, buffer_size=30):
        self.buffer_size = buffer_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self):
        current_time = time.time()
        
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        
        if len(self.frame_times) > self.buffer_size:
            self.frame_times.pop(0)
    
    def get_fps(self):
        if len(self.frame_times) < 2:
            return 0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)

        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0
