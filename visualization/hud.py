import cv2
import numpy as np

class HUD:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        
    def render(self, frame, explained_frame=None, controls=None, fps=0, xai_method="None", safety_score=100.0):
        """
        frame: Original camera frame (BGR)
        explained_frame: Frame with heatmap overlay (optional)
        controls: (steering, throttle, brake) tuple
        safety_score: Driving component score (0-100)
        """
        # Resize frame to HUD size (or keep aspect ratio)
        if frame is not None:
            # If explained_frame present, maybe show side-by-side or picture-in-picture
            # Or just use the explained frame as main info.
            
            if explained_frame is not None:
                display_img = explained_frame
            else:
                display_img = frame
                
            display_img = cv2.resize(display_img, (self.width, self.height))
            
            # Draw UI
            # Info box
            cv2.rectangle(display_img, (10, 10), (300, 250), (0, 0, 0), -1)
            cv2.putText(display_img, f"XAI Method: {xai_method}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_img, f"FPS: {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if controls:
                steer, throttle, brake = controls
                cv2.putText(display_img, f"Steer: {steer:.2f}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_img, f"Throttle: {throttle:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_img, f"Brake: {brake:.2f}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
                # Visual bars
                self._draw_bar(display_img, steer, 20, 145, color=(0, 255, 0), center=True)
                self._draw_bar(display_img, throttle, 20, 165, color=(0, 255, 255))
                self._draw_bar(display_img, brake, 20, 185, color=(0, 0, 255))

            # Safety Score
            score_color = (0, 255, 0) if safety_score > 80 else (0, 165, 255) if safety_score > 50 else (0, 0, 255)
            cv2.putText(display_img, f"Safety Score: {int(safety_score)}%", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
            self._draw_bar(display_img, safety_score/100.0, 20, 230, width=200, height=10, color=score_color)

            return display_img
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
    def _draw_bar(self, img, value, x, y, width=200, height=10, color=(255, 255, 255), center=False):
        cv2.rectangle(img, (x, y), (x + width, y + height), (50, 50, 50), -1)
        
        if center:
            # Value from -1 to 1. Center is x + width/2
            bar_w = int((value) * (width / 2))
            center_x = x + width // 2
            cv2.rectangle(img, (center_x, y), (center_x + bar_w, y + height), color, -1)
            cv2.line(img, (center_x, y-2), (center_x, y+height+2), (255, 255, 255), 1)
        else:
            # Value from 0 to 1
            bar_w = int(value * width)
            cv2.rectangle(img, (x, y), (x + bar_w, y + height), color, -1)
