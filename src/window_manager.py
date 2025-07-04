import cv2 as cv

class WindowManager:
    def __init__(self, window_title, camera_width=640, camera_height=480):
        self.window_title = window_title
        self.current_width = camera_width
        self.current_height = camera_height
        self.scale_factor = 1.0
        self.last_window_size = (camera_width, camera_height)

        # Create window with resize capability
        cv.namedWindow(self.window_title, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(self.window_title, self.current_width, self.current_height)
        
        # Set minimum window size
        cv.setWindowProperty(self.window_title, cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_FREERATIO)
    
    def update_window_size(self):
        """Check if window was resized and update internal state"""
        try:
            # Get current window size
            window_rect = cv.getWindowImageRect(self.window_title)
            if window_rect != (-1, -1, -1, -1):  # Valid window
                new_width, new_height = window_rect[2], window_rect[3]
                
                # Check if size changed significantly
                if abs(new_width - self.current_width) > 5 or abs(new_height - self.current_height) > 5:
                    self.current_width = new_width
                    self.current_height = new_height
                    
                    # Calculate scale factor based on original camera resolution
                    scale_x = new_width / self.last_window_size[0]
                    scale_y = new_height / self.last_window_size[1]
                    self.scale_factor = min(scale_x, scale_y)  # Maintain aspect ratio
                                        
                    return True
        except:
            pass
        
        return False
    
    def resize_frame(self, frame):
        """Resize frame to fit current window size while maintaining aspect ratio"""
        if self.scale_factor == 1.0:
            return frame
        
        # Calculate new dimensions
        height, width = frame.shape[:2]
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        
        # Resize frame
        resized_frame = cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_LINEAR)
        
        return resized_frame
    
    def scale_coordinates(self, coordinates):
        """Scale coordinates to match current window size"""
        if self.scale_factor == 1.0:
            return coordinates
        
        scaled_coords = []
        for coord in coordinates:
            if hasattr(coord, 'left'):  # dlib rectangle
                scaled_rect = type(coord)(
                    int(coord.left() * self.scale_factor),
                    int(coord.top() * self.scale_factor),
                    int(coord.right() * self.scale_factor),
                    int(coord.bottom() * self.scale_factor)
                )
                scaled_coords.append(scaled_rect)
            else:  # tuple or list
                scaled_coords.append(tuple(int(c * self.scale_factor) for c in coord))
        
        return scaled_coords
    
    def get_scale_factor(self):
        return self.scale_factor
