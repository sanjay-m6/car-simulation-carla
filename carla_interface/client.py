import glob
import os
import sys
import time
import queue
import numpy as np
import cv2

try:
    import carla
except ImportError:
    # Try to find carla egg if not installed via pip
    # This is a common pattern in CARLA scripts
    try:
        sys.path.append(glob.glob('**/carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
        import carla
    except:
        print("CARLA module not found. Please ensure CARLA is installed or egg is in path.")
        carla = None

class CarlaClient:
    def __init__(self, host='127.0.0.1', port=2000):
        if carla is None:
            raise RuntimeError(
                "CARLA Python module is not installed or not found. \n"
                "Please install the CARLA simulator and its Python API: \n"
                "  1. Download CARLA from https://github.com/carla-simulator/carla/releases \n"
                "  2. Install the Python API: pip install <carla_root>/PythonAPI/carla/dist/carla-<version>.whl \n"
                "Or, ensure the CARLA server is running and the egg file is in your path."
            )
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle = None
        self.camera = None
        self.image_queue = queue.Queue()
        
    def setup_vehicle(self):
        # Spawn a vehicle
        bp = self.blueprint_library.filter('model3')[0]
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(bp, spawn_point)
        self.vehicle.set_autopilot(True) # Start with autopilot or manual? 
        # Usually for inference we control it.
        # self.vehicle.set_autopilot(False)
        print("Vehicle spawned.")
        
    def setup_camera(self):
        # Spawn a camera
        bp = self.blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '640') # Capture at higher res
        bp.set_attribute('image_size_y', '480')
        bp.set_attribute('fov', '110')
        
        # Position: hood/roof
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        
        # Listen to data
        self.camera.listen(self.image_queue.put)
        print("Camera spawned and listening.")
        
    def get_data(self):
        # Get latest image
        try:
            data = self.image_queue.get(timeout=2.0)
            
            # Convert to numpy
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4)) # BGRA
            array = array[:, :, :3] # BGR
            
            # Use OpenCV to handle BGR -> RGB later if needed
            return array, data.frame
        except queue.Empty:
            return None, None
            
    def apply_control(self, steering, throttle, brake):
        control = carla.VehicleControl()
        control.steer = float(steering)
        control.throttle = float(throttle)
        control.brake = float(brake)
        self.vehicle.apply_control(control)
        
    def cleanup(self):
        if self.camera:
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        print("Actors destroyed.")
