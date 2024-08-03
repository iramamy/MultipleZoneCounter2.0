import cv2

class Color:
    def __init__(self):
        # Predefined colors
        self.colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
            (192, 192, 192),
            (255, 42, 4),
        ]
    
    def __getitem__(self, index):
        if 0 <= index < len(self.colors):
            return self.colors[index]
        else:
            raise IndexError("Color index out of range")

    def add_color(self, color):
        if len(color) == 3 and all(0 <= c <= 255 for c in color):
            self.colors.append(color)
        else:
            raise ValueError("Color must be a tuple of 3 values between 0 and 255")

def draw_polygon(
    zones_list,
    polygon_colors,
    frame,
    line_thickness=2,
    point_thickness=2
    ):

    color_used = []
    for i, zone in enumerate(zones_list):
        if len(zone) > 0:

            # Cycle through colors if more zones than colors
            color = polygon_colors[i % len(polygon_colors.colors)]
            color_used.append(color)

            for point in zone:
                cv2.circle(frame, point, point_thickness, color, -1)
            
            for j in range(len(zone) - 1):
                cv2.line(frame, zone[j], zone[j + 1], color, line_thickness)
            
            # Connect the last point to the first to complete the shape if drawing is finished
            if len(zone) > 1:
                cv2.line(frame, zone[-1], zone[0], color, line_thickness)
    
    return color_used