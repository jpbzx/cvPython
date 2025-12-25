import math

class DistanceFinder:
    def calculate_distance(self, x1, y1, x2, y2):
        distance =  math.sqrt((x2 - x1) ** 2 + (y2 - y1) **2)

        if distance > 0 and distance < 100:
            return distance
        elif distance > 100:
            distance = 100
            return distance