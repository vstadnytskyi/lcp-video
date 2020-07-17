
from time import time, sleep
from numpy import zeros, right_shift, array, copy
import PySpin

from PySpin import PixelFormat_Mono16,PixelFormat_Mono12Packed

class FlirCamera():

    def __init__(self, name = None, system = None):
        """
        """
        from numpy import zeros, nan
        self.name = name
        self.system = system

        #Acquisition
        self.queue_length = 64
        self.acquiring = False
        self.header_length = 4096

        #Recording
        self.recording_filename = f'camera_{name}.hdf5'
        self.recording_root = '/mnt/data/'
        self.recording_N = 1
        self.recording = False
        self.recording_pointer = 0
        self.recording_chunk_pointer = 0
        self.recording_chunk_maxpointer = 1024

        self.threads = {}




    def init(self, serial_number, settings = 1):
        from numpy import zeros, nan
        print('')
        print('-------------INITIALIZING NEW CAMERA-----------')
        import PySpin
        from circular_buffer_numpy.queue import Queue


        self.cam = self.find_camera(serial_number = serial_number)
        self.nodes = self.get_nodes()

        self.height = self.get_height()
        self.width = self.get_width()
        self.img_len = int(self.height*self.width*1.5)
        self.queue = Queue((self.queue_length,self.img_len+self.header_length), dtype = 'uint8')
        self.last_image = zeros((self.img_len+self.header_length,), dtype = 'uint8')
        #Algorithms Configuration
        self.lut_enable = False
        self.gamma_enable = False
        self.autoexposure = 'Off'
        self.autogain = 'Off'

        #Transport Configuration
        self.configure_transport()

        #Analog Configuration
        self.conf_acq_and_trigger(settings=settings)

        try:
            self.cam.AcquisitionStop()
            print ("Acquisition stopped")
        except PySpin.SpinnakerException as ex:
            print("Acquisition was already stopped")



        self.set_exposure_mode('Timed')
        self.exposure_time = 63000 #53500
        self.gain = 0
        self.black_level = 15


        self.background = zeros((self.height+1,self.width))
        self.background[:self.height,:] = 15
        self.background_flag = True



    def close(sself):
        pass

    def kill(self):
        """
        """
        self.pause_acquisition()
        sleep(1)
        self.stop_acquisition()
        sleep(0.1)
        del self.cam
        self.system.ReleaseInstance()
        del self

    def reset_to_factory_settings(self):
        self.pause_acquisition()
        self.stop_acquisition()
        try:
            self.cam.AcquisitionStop()
            print ("Acquisition stopped")
        except PySpin.SpinnakerException as ex:
            print("Acquisition was already stopped")
        self.cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
        self.cam.UserSetLoad()

    def configure_transport(self):
        cam = self.cam
        # Configure Transport Layer Properties
        cam.TLStream.StreamBufferHandlingMode.SetValue(PySpin.StreamBufferHandlingMode_OldestFirst )
        cam.TLStream.StreamBufferCountMode.SetValue(PySpin.StreamBufferCountMode_Manual )
        cam.TLStream.StreamBufferCountManual.SetValue(10)
        print(f"Buffer Handling Mode: {cam.TLStream.StreamBufferHandlingMode.GetCurrentEntry().GetSymbolic()}")
        print( f"Buffer Count Mode: {cam.TLStream.StreamBufferCountMode.GetCurrentEntry().GetSymbolic()}")
        print(f"Buffer Count: {cam.TLStream.StreamBufferCountManual.GetValue()}")
        print(f"Max Buffer Count: {cam.TLStream.StreamBufferCountManual.GetMax()}")

    def deinit(self):
        self.stop_acquisition()
        self.cam.DeInit()


    def find_camera(self, serial_number = None):
        """
        looks for all cameras connected and returns cam object
        """
        cam = None
        cam_list = self.system.GetCameras()
        num_cameras = cam_list.GetSize()
        print(f'found {num_cameras} cameras')
        for i,cam in enumerate(cam_list):
            sn = cam.TLDevice.DeviceSerialNumber.GetValue()
            print(f'sn = {sn}')
            if serial_number == sn:
                break
        cam_list.Clear()
        return cam

    def get_all_cameras(self):
        cam_list = self.system.GetCameras()
        num_cameras = cam_list.GetSize()
        print(f'found {num_cameras} cameras')
        cameras = []
        for i,cam in enumerate(cam_list):
            sn = cam.TLDevice.DeviceSerialNumber.GetValue()
            print(f'sn = {sn}')
            cameras.append(sn)
        cam_list.Clear()
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


    def get_background(self):
        from numpy import mean
        self.background = mean(self.queue.buffer, axis = 0)

    def get_image(self):
        self.last_image *= 0
        if self.acquiring:
            image_result = self.cam.GetNextImage()
            timestamp = image_result.GetTimeStamp()
            frameid = image_result.GetFrameID()
            # Getting the image data as a numpy array
            image_data = image_result.GetData()
            image_result.Release()
        else:
            print('No Data in get image')
            image_data = zeros((self.height*self.width,))
        pointer = self.img_len
        self.last_image[:pointer] = image_data
        self.last_image[pointer:pointer+64] = self.get_image_header(value =int(time()*1000000), length = 64)
        self.last_image[pointer+64:pointer+128] = self.get_image_header(value = timestamp, length = 64)
        self.last_image[pointer+128:pointer+192] = self.get_image_header(value = frameid, length = 64)
        return self.last_image

    def get_image_header(self,value = None, length = 4096):
        from time import time
        from numpy import zeros
        from ubcs_auxiliary.numerical import bin_array
        arr = zeros((1,length))
        if value is None:
            t = int(time()*1000000)
        else:
            t = value
        arr[0,:64] = bin_array(t,64)
        return arr

    def binarr_to_number(self,arr):
        num = 0
        from numpy import flip
        arr = flip(arr)
        length = arr.shape[0]
        for i in range(length):
           num += (2**(i))*arr[i]
        return num

    def bin_array(num, m):
        from numpy import uint8, binary_repr,array
        """Convert a positive integer num into an m-bit bit vector"""
        return array(list(binary_repr(num).zfill(m))).astype(uint8)

    def run_once(self):
        if self.acquiring:
            img = self.get_image()
            data = img.reshape(1,self.img_len+4096)
            self.queue.enqueue(data)

    def run(self):
        while self.acquiring:
            self.run_once()


    def start_thread(self):
        from ubcs_auxiliary.multithreading import new_thread
        if not self.acquiring:
            self.start_acquisition()
            self.threads['acquisition'] = new_thread(self.run)

    def stop_thread(self):
        if self.acquiring:
            self.stop_acquisition()

    def resume_acquisition(self):
        """
        """
        from ubcs_auxiliary.multithreading import new_thread
        if not self.acquiring:
            self.acquiring = True
            self.threads['acquisition'] = new_thread(self.run)

    def pause_acquisition(self):
        """
        """
        self.acquiring = False

    def start_acquisition(self):
        """
        a wrapper to start acquisition of images.
        """
        self.acquiring = True
        self.cam.BeginAcquisition()
    def stop_acquisition(self):
        """
        a wrapper to stop acquisition of images.
        """
        self.acquiring = False
        try:
            self.cam.EndAcquisition()
            print ("Acquisition ended")
        except PySpin.SpinnakerException as ex:
            print("Acquisition was already ended")


    def get_black_level(self):
        self.cam.BlackLevelSelector.SetValue(0)
        all = self.cam.BlackLevel.GetValue()*4095/100
        self.cam.BlackLevelSelector.SetValue(1)
        analog = self.cam.BlackLevel.GetValue()*4095/100
        self.cam.BlackLevelSelector.SetValue(2)
        digital = self.cam.BlackLevel.GetValue()*4095/100
        self.cam.BlackLevelSelector.SetValue(0)
        return {'all':all,'analog':analog,'digital':digital}

    def set_black_level(self,value):
        """
        """
        self.cam.BlackLevelSelector.SetValue(0)
        self.cam.BlackLevel.SetValue(value*100/4095)
    black_level = property(get_black_level,set_black_level)

    def get_temperature(self):
        """
        """
        temp = self.cam.DeviceTemperature.GetValue()
        return temp
    temperature = property(get_temperature)

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
        return device_serial_number

    def get_acquisition_mode(self):
        node = self.nodes['acquisition_mode']

    def set_acquisition_mode(self,value):
        node = self.nodes['acquisition_mode']
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

    def get_lut_enable(self):
        """
        """
        if self.cam is not None:
            result = self.cam.lut_enable.GetValue()
        else:
            result = None
        return result
    def set_lut_enable(self,value):
        """
        """
        if self.cam is not None:
            print('setting Look Up Table Enable to False')
            self.cam.LUTEnable.SetValue(value)
    lut_enable = property(get_lut_enable,set_lut_enable)

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

    def get_gamma(self):
        if self.cam is not None:
            result = None
            #result = self.cam.Gamma.GetValue()
        else:
            result = None
        return result
    def set_gamma(self,value):
        """
        """
        if self.cam is not None:
            result = None
            #self.cam.Gamma.SetValue(value)
        print(f'setting gamma {value}')
    gamma = property(get_gamma,set_gamma)

    def get_height(self):
        if self.cam is not None:
            reply = self.cam.Height.GetValue()
        else:
            reply = nan
        return reply

    def get_width(self):
        if self.cam is not None:
            reply = self.cam.Width.GetValue()
        else:
            reply = nan
        return reply

    def set_exposure_mode(self,mode = None):
        from PySpin import ExposureMode_Timed, ExposureMode_TriggerWidth
        print(f'Setting up ExposureMode to {mode}')
        if self.cam is not None:
            if mode == "Timed":
                self.cam.ExposureMode.SetValue(ExposureMode_Timed)
            elif mode == "TriggerWidth":
                self.cam.ExposureMode.SetValue(ExposureMode_TriggerWidth)

    def set_autoexposure(self, value = 'Off'):
        from PySpin import ExposureAuto_Off, ExposureAuto_Continuous, ExposureAuto_Once
        if value == 'Off':
            self.cam.ExposureAuto.SetValue(ExposureAuto_Off)
        elif value == 'Once':
            self.cam.ExposureAuto.SetValue(ExposureAuto_Once)
        elif value == 'Continuous':
            self.cam.ExposureAuto.SetValue(ExposureAuto_Continuous)
            print(f'setting gamma enable {value}')
    def get_autoexposure(self):
        value = self.cam.ExposureAuto.GetValue()
        if value == 0:
            return 'Off'
        elif value == 1:
            return 'Once'
        elif value == 2:
            return 'Continuous'
        else:
            return 'unknown'
    autoexposure = property(get_autoexposure,set_autoexposure)

    def set_exposure_time(self, value):
        if self.cam is not None:
            self.cam.ExposureTime.SetValue(value)
            self._exposure_time = value
        else:
            pass
    def get_exposure_time(self,):
        if self.cam is not None:
            return self.cam.ExposureTime.GetValue()
        else:
            return nan
        self._exposure_time = value
    exposure_time = property(get_exposure_time,set_exposure_time)

    def trigger_now(self):
        """
        software trigger for camera
        """
        self.cam.TriggerSoftware()

    def conf_acq_and_trigger(self, settings = 1):
        """
        a collection of setting defined as appropriate for the selected modes of operation.

        settings 1 designed for external trigger mode of operation
        Acquisition Mode = Continuous
        Acquisition FrameRate Enable = False
        TriggerSource = Line 0
        TriggerMode = On
        TriggerSelector = FrameStart
        """
        import PySpin

        "Two different pixel formats: PixelFormat_Mono12p and PixelFormat_Mono12Packed"
        print('setting pixel format Mono12Packed')
        self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono12Packed)

        if self.cam is not None:
            if settings ==1:
                print('Acquisition and Trigger Settings: 1')
                print('setting continuous acquisition mode')
                self.cam.AcquisitionMode.SetIntValue(PySpin.AcquisitionMode_Continuous)
                print('setting frame rate enable to False')
                self.cam.AcquisitionFrameRateEnable.SetValue(False)



                print('setting TriggerSource Line0')
                self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
                print('setting TriggerMode On')
                self.cam.TriggerMode.SetIntValue(PySpin.TriggerMode_On)

                print('setting TriggerSelector FrameStart')
                self.cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
                print('setting TriggerActivation RisingEdge')
                self.cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
                print('setting TriggerOverlap ReadOnly ')
                self.cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)


            elif settings ==2:
                print('Acquisition and Trigger Settings: 2')
                print('setting SingleFrame acquisition mode')
                self.cam.AcquisitionMode.SetIntValue(PySpin.AcquisitionMode_SingleFrame)
                print('setting frame rate enable to False')
                self.cam.AcquisitionFrameRateEnable.SetValue(False)

                print('setting frame rate enable to False')
                self.cam.AcquisitionFrameRateEnable.SetValue(False)
                print('setting TriggerMode On')
                self.cam.TriggerMode.SetIntValue(PySpin.TriggerMode_Off)


            elif settings == 3:
                print('Acquisition and Trigger Settings: 3 (software)')
                print('setting continuous acquisition mode')
                self.cam.AcquisitionMode.SetIntValue(PySpin.AcquisitionMode_Continuous)

                print('setting frame rate enable to True')
                self.cam.AcquisitionFrameRateEnable.SetValue(True)
                print('setting frame rate to 1')
                self.cam.AcquisitionFrameRate.SetValue(20.0)
                print('setting TriggerMode Off')
                self.cam.TriggerMode.SetIntValue(PySpin.TriggerMode_Off)
                print('setting TriggerSource Line0')
                self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
                print('setting TriggerSource Software')
                self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)


# Recording

    def recording_init(self, N_frames = 1200, comments = ''):
        """
        Initializes recording
        """
        self.recording_basefilename = self.recording_root+f'{self.name}_{comments}'
        self.recording_chunk_pointer = 0
        filename = self.recording_basefilename + '_' + str(self.recording_chunk_pointer) + '.tmpraw.hdf5'
        self.recording_create_file(filename = filename, N_frames = N_frames)
        self.recording_Nframes = N_frames
        self.recording_pointer = 0

    def recording_create_file(self, filename, N_frames):
        """
        creates hdf5 file on a drive prior to recording
        """
        from os.path import exists
        from h5py import File
        from time import time, sleep, ctime
        from numpy import zeros
        if exists(filename):
            print('ctime(time()): ----WARNING----')
            print('ctime(time()): The HDF5 file exists. Please delete it first!')
        else:
            print(f'{ctime(time())}: The HDF5 was created. The file name is {filename}')
            with File(filename,'a') as f:
                f.create_dataset('exposure time', data = 'trigger width')
                f.create_dataset('black level all', data = self.black_level['all'])
                f.create_dataset('black level analog', data = self.black_level['analog'])
                f.create_dataset('black level digital', data = self.black_level['digital'])
                f.create_dataset('image width', data = self.width)
                f.create_dataset('image height', data = self.height)
                f.create_dataset('gain', data = self.gain)
                f.create_dataset('time', data = ctime(time()))
                f.create_dataset('temperature', data = self.temperature)
                f.create_dataset('images', (N_frames,self.img_len), dtype = 'uint8', chunks = (1,self.img_len))
                f.create_dataset('timestamps_lab', (N_frames,) , dtype = 'float64')
                f.create_dataset('timestamps_camera', (N_frames,) , dtype = 'float64')
                f.create_dataset('frameIDs', (N_frames,) , dtype = 'float64')
        self.recording_filename = filename

    def record_once(self,filename,N):
        """
        records a sequence of N frames.
        """
        from h5py import File
        images = self.queue.dequeue(N)
        if images.shape[0] > 0:
            with File(filename,'a') as f:
                for i in range(N):
                    pointer = self.recording_pointer
                    image = images[i]
                    tlab,tcam,frameID = self.extract_timestamp_image(image)
                    f['images'][pointer] = image[:self.img_len]
                    f['timestamps_lab'][pointer] = tlab
                    f['timestamps_camera'][pointer] = tcam
                    f['frameIDs'][pointer] = frameID
                    self.recording_pointer += 1
        else:
            print(f'{self.name}: got empty array from deque with rear value in the queue of {self.queue.rear}')

    def recording_run(self):
        """
        records iamges to file in a loop.
        """
        from time import time,ctime
        from os import utime, rename
        from h5py import File
        while (self.recording):
            if self.recording_pointer >= self.recording_Nframes-1:
                self.recording_chunk_pointer += 1
                if self.recording_chunk_pointer >= self.recording_chunk_maxpointer:
                    self.recording = False
                    break
                self.recording_pointer = 0
                rename(self.recording_basefilename + '_' + str(self.recording_chunk_pointer-1) + '.tmpraw.hdf5',self.recording_basefilename + '_' + str(self.recording_chunk_pointer-1) + '.raw.hdf5')
                f = File(self.recording_basefilename + '_' + str(self.recording_chunk_pointer-1) + '.raw.hdf5','r')
                timestamp = f['timestamps_lab'][0]
                utime(self.recording_basefilename + '_' + str(self.recording_chunk_pointer-1) + '.raw.hdf5',(timestamp,timestamp))

                filename = self.recording_basefilename + '_' + str(self.recording_chunk_pointer) + '.tmpraw.hdf5'
                Nframes = self.recording_Nframes
                self.recording_create_file(filename,Nframes)

            filename = self.recording_filename
            N = self.recording_N
            if (self.queue.length > N):
                self.record_once(filename = filename, N = N)
            else:
                sleep((N+3)*0.051)
        self.recording = False
        print(ctime(time()),f'Recording Loop is finished')

    def recording_start(self):
        """
        a simple wrapper to start a recording_run loop in a separate thread.
        """
        from ubcs_auxiliary.multithreading import new_thread
        from time import time, sleep
        self.recording = True
        self.threads['recording'] = new_thread(self.recording_run)

    def recording_stop(self):
        """
        a wrapper that stops the recording loop
        """
        self.recording = False
        self.threads['recording'] = None

    def extract_timestamp_image(self,img):
        """
        Extracts header information from the
        """
        pointer = self.img_len
        header_img = img[pointer:pointer+64]
        timestamp_lab =  self.binarr_to_number(header_img)/1000000
        header_img = img[pointer+64:pointer+128]
        timestamp_cam =  self.binarr_to_number(header_img)
        header_img = img[pointer+128:pointer+192]
        frameid =  self.binarr_to_number(header_img)
        return timestamp_lab,timestamp_cam,frameid

    def binarr_to_number(self,vector):
        """
        converts a vector of bits into an integer.
        """
        num = 0
        from numpy import flip
        vector = flip(vector)
        length = vector.shape[0]
        for i in range(length):
           num += (2**(i))*vector[i]
        return num

    def bin_array(self,num, m):
        """
        Converts a positive integer num into an m-bit bit vector
        """
        from numpy import uint8, binary_repr,array
        return array(list(binary_repr(num).zfill(m))).astype(uint8)


    def get_conversion_mask(self):
        from numpy import vstack, tile, hstack, arange
        length = int(self.width*self.height*1.5)
        b0 = 2**hstack((arange(4,12,1),arange(4)))
        b1 = 2**arange(12)
        b = vstack((b0,b1))
        bt = tile(b,(int((length/(2*1.5))),1)).astype('uint16')
        return bt

    def raw_to_image(self, rawdata, height = None, width = None, mask = None):
        from numpy import vstack, tile, hstack, arange,reshape
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        if mask is None:
            mask = self.get_conversion_mask()
        data_Nx8 = ((rawdata.reshape((-1,1)) & (2**arange(8))) != 0)
        data_N8x1 = data_Nx8.flatten()
        data_Mx12 = data_N8x1.reshape((int(rawdata.shape[0]/1.5),12))
        data = (data_Mx12*mask).sum(axis=1)
        return data.reshape((height,width)).astype('int16')

if __name__ is '__main__':
    from PySpin import System
    system = System.GetInstance()
    camera = FlirCamera('test',system)
    camera.get_all_cameras()

    print("camera_dm4 = FlirCamera('dm4', system)")
    print("camera_dm4.init(serial_number = '19490369')")
    print('camera_dm4.start_thread()')

    print("camera_dm34 = FlirCamera('dm34',system)")
    print("camera_dm34.init(serial_number = '18159480')")
    print('camera_dm34.start_thread()')

    print("camera_dm16 = FlirCamera('dm16',system)")
    print("camera_dm16.init(serial_number = '18159488')")
    print('camera_dm16.start_thread()')

    print("camera_out = FlirCamera('dmout',system)")
    print("camera_out.init(serial_number = '20130136')")
    print('camera_out.start_thread()')
    from matplotlib import pyplot as plt
    plt.ion()
