
from time import time, sleep
from numpy import zeros, right_shift
import PySpin

class FlirCamera():

    def __init__(self):
        self.last_image_time = zeros((10,))

    def init(self, serial_number):
        import PySpin
        from circular_buffer_numpy.circular_buffer import CircularBuffer
        self.acquiring = False

        self.find_camera(serial_number = serial_number)
        self.nodes = self.get_nodes()
        self.height = self.get_height()
        self.width = self.get_width()
        self.buffer = CircularBuffer((10,self.height,self.width), dtype = 'uint16')

        self.set_gamma_enable(False)
        self.set_autoexposure('Off')
        self.set_autogain('Off')

        print('setting pixel format Mono12Packed')
        #self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono12Packed)

        print('setting continuous acquisition mode')
        self.cam.AcquisitionMode.SetIntValue(PySpin.AcquisitionMode_Continuous)

        print('setting frame rate enable to True')
        self.cam.AcquisitionFrameRateEnable.SetValue(True)
        print('setting frame rate to 1')
        self.cam.AcquisitionFrameRate.SetValue(10.0)

        self.set_exposure_mode('Timed')
        self.set_exposure_time(10*1000)

        self.conf_trigger()

        from numpy import zeros


    def deinit(self):
        self.stop_acquisition()
        self.cam.DeInit()
        del self.cam
        self.system.ReleaseInstance()


    def find_camera(self, serial_number = None):
        """
        looks for all cameras connected and returns cam object
        """
        from PySpin import System
        self.system = System.GetInstance()
        cam_list = self.system.GetCameras()
        num_cameras = cam_list.GetSize()
        print(f'found {num_cameras} cameras')
        for i,cam in enumerate(cam_list):
            sn = cam.TLDevice.DeviceSerialNumber.GetValue()
            print(f'sn = {sn}')
            if serial_number == sn:
                self.cam = cam
                break
            else:
                self.cam = None
        cam_list.Clear()

    def get_all_cameras(self):
        from PySpin import System
        self.system = System.GetInstance()
        cam_list = self.system.GetCameras()
        num_cameras = cam_list.GetSize()
        print(f'found {num_cameras} cameras')
        cameras = []
        for i,cam in enumerate(cam_list):
            sn = cam.TLDevice.DeviceSerialNumber.GetValue()
            print(f'sn = {sn}')
            cameras.append(sn)
        return cameras

    def get_nodes(self):
        import PySpin
        self.cam.Init()
        # Retrieve GenICam nodemap
        nodemap = self.cam.GetNodeMap()
        self.nodemap = nodemap
        nodes = {}
        nodes['auto_gain'] = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        nodes['pixel_format'] = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
        nodes['gain'] = PySpin.CEnumerationPtr(nodemap.GetNode('Gain'))
        nodes['acquisition_mode'] = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        nodes['exposure_time'] = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureTime'))
        return nodes

    def get_image(self):
        from numpy import right_shift
        from PySpin import PixelFormat_Mono16
        if self.acquiring:
            self.last_image_time[3] = time()-self.t_start
            image_result = self.cam.GetNextImage()
            self.last_image_time[4] = time()-self.t_start
            # Getting the image data as a numpy array
            image_data = image_result.Convert(PixelFormat_Mono16).GetNDArray()
            self.last_image_time[5] = time()-self.t_start
            image_result.Release()
            self.last_image_time[6] = time()-self.t_start
        else:
            image_data = None
        self.last_image_time[7] = time()-self.t_start
        return image_data

    def run_once(self):
        if self.acquiring:
            self.t_start = time()
            img = self.get_image()
            self.last_image_time[8] = time()-self.t_start
            self.buffer.append((img.reshape(1,img.shape[0],img.shape[1])))
            self.last_image_time[9] = time()-self.t_start

    def run(self):
        while self.acquiring:
            self.run_once()

    def start_thread(self):
        from ubcs_auxiliary.threading import new_thread
        self.start_acquisition()
        new_thread(self.run)

    def stop_thread(self):
        self.acquiring = False
        self.stop_acquisition()

    def start_acquisition(self):
        """
        a wrapper to start acquisition of images.
        """
        self.acquiring = True
        self.cam.BeginAcquisition()

    def stop_acquisition(self):
        """
        a wrapper to sstop acquisition of images.
        """
        self.acquiring = False
        self.cam.EndAcquisition()

    def get_fps():
        return nan
    def set_fps(value):
        pass
    fps = property(get_fps,set_fps)

    def get_gain(self):
        return self.cam.Gain.GetValue()
    def set_gain(self,value):
        if value >= self.cam.Gain.GetMax():
            value = self.cam.Gain.GetMax()
        elif value < self.cam.Gain.GetMin() :
            value = self.cam.Gain.GetMin()
        self.cam.Gain.SetValue(value)
    gain = property(get_gain,set_gain)

    def get_autogain(self):
        if self.cam is not None:
            value = self.cam.GainAuto.GetValue()
            if value == 0:
                return 'Off'
            elif value == 1:
                return 'Once'
            elif value == 2:
                return 'Continuous'
        else:
            return None
    def set_autogain(self,value):
        node = self.nodes['auto_gain']
        off = node.GetEntryByName("Off")
        once = node.GetEntryByName("Once")
        continuous = node.GetEntryByName("Continuous")
        if value == 'Off':
            node.SetIntValue(off.GetValue())
        elif value == 'Once':
            node.SetIntValue(once.GetValue())
        elif value == 'Continuous':
            node.SetIntValue(continuous.GetValue())
    autogain = property(get_autogain,set_autogain)

    def get_serial_number(self):
        import PySpin
        device_serial_number = PySpin.CStringPtr(self.nodemap_tldevice.GetNode('DeviceSerialNumber'))

    def get_acquisition_mode(self):
        node = self.nodes['acquisition_mode']

    def set_acquisition_mode(self,value):
        import PySpin
        continious = node.GetEntryByName("Continuous")
        single_frame = node.GetEntryByName("SingleFrame")
        multi_frame = node.GetEntryByName("MultiFrame")
        if value == 'Continuous':
            mode = continious
        elif value == 'SingleFrame':
            mode = single_frame
        elif value == 'MultiFrame':
            mode = multi_frame
        print(f'setting acquisition mode {value}')
        self.stop_acquisition()
        node.SetIntValue(mode)
        self.start_acquisition()

    def get_gamma_enable(self):
        """
        """
        if self.cam is not None:
            result = self.cam.GammaEnable.GetValue()
        else:
            result = None
        return result
    def set_gamma_enable(self,value):
        """
        """
        if self.cam is not None:
            self.cam.GammaEnable.SetValue(value)
        print(f'setting gamma enable {value}')
    gamma_enable = property(get_gamma_enable,set_gamma_enable)

    def get_height(self):
        if self.cam is not None:
            reply = self.cam.Height.GetValue()
        else:
            replt = None
        return reply

    def get_width(self):
        if self.cam is not None:
            reply = self.cam.Width.GetValue()
        else:
            reply = None
        return reply

    def set_exposure_mode(self,mode = None):
        from PySpin import ExposureMode_Timed
        if self.cam is not None:
            if mode == "Timed":
                self.cam.ExposureMode.SetValue(ExposureMode_Timed)

    def set_autoexposure(self, value = 'Off'):
        from PySpin import ExposureAuto_Off, ExposureAuto_Continuous, ExposureAuto_Once
        if value == 'Off':
            self.cam.ExposureAuto.SetValue(ExposureAuto_Off)
        elif value == 'Once':
            self.cam.ExposureAuto.SetValue(ExposureAuto_Once)
        elif value == 'Continuous':
            self.cam.ExposureAuto.SetValue(ExposureAuto_Continuous)
            print(f'setting gamma enable {value}')
    def get_autoexposure(self, bool = False):
        value = self.cam.ExposureAuto.GetValue()
        if value == 0:
            return 'Off'
        elif value == 1:
            return 'Once'
        elif value == 2:
            return 'Continuous'
        else:
            return 'unknown'

    def set_exposure_time(self, value):
        if self.cam is not None:
            self.cam.ExposureTime.SetValue(value)
        else:
            pass
    def get_exposure_time(self,):
        if self.cam is not None:
            return self.cam.ExposureTime.GetValue()
        else:
            return nan
    exposure_time = property(get_exposure_time,set_exposure_time)

    def set_pixel_format(self, value = 'Mono16'):
        self.cam.PixelFormat.SetIn

##### Acquisition Control
#####


    def trigger_now(self):
        """
        software trigger for camera
        """
        import PySpin
        self.cam.TriggerSoftware()

    def conf_trigger(self):
        import PySpin
        if self.cam is not None:
            print('setting TriggerMode Off')
            self.cam.TriggerMode.SetIntValue(PySpin.TriggerMode_Off)

            print('setting TriggerSource Line0')
            self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)


            # print('setting TriggerMode On')
            # self.cam.TriggerMode.SetIntValue(PySpin.TriggerMode_On)
            #
            # print('setting TriggerSelector FrameStart')
            # self.cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
            # print('setting TriggerSource Software')
            # self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)

            #print('setting TriggerActivation RisingEdge')
            #this works only if Trigger is not Software
            #self.cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)

def get_gain_mean_std(camera):
    from numpy import array
    from time import sleep
    lst = []
    for g in range(0,49):
        camera.gain = g
        gain = camera.gain
        camera.trigger_now();
        sleep(0.3)
        img = camera.buffer.get_last_value()[0]
        print(gain,img.mean(),img.std())
        lll = [gain,img.mean(),img.std()]
        lst.append(lll)
    return array(lst)

if __name__ is '__main__':
    camera = FlirCamera()
    camera.get_all_cameras()

    print("camera_tc = FlirCamera()")
    print("camera_tc.init(serial_number = '19490369')")
    print('camera_tc.start_thread()')

    print("camera_12mm = FlirCamera()")
    print("camera_12mm.init(serial_number = '18159480')")
    print('camera_12mm.start_thread()')

    print("camera_8mm = FlirCamera()")
    print("camera_8mm.init(serial_number = '18159488')")
    print('camera_8mm.start_thread()')

    from matplotlib import pyplot as plt
    plt.ion()
