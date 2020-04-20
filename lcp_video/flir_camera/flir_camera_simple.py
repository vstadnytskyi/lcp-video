import PySpin

system = PySpin.System.GetInstance()

cam_list = system.GetCameras()

num_cameras = cam_list.GetSize()
print('Number of cameras detected: %d' % num_cameras)

# Finish if there are no cameras
if num_cameras == 0:

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    print('Not enough cameras!')
    input('Done! Press Enter to exit...')

# Run example on each camera

cameras = []
for i, cam in enumerate(cam_list):
    cameras.append([i,cam])

if num_cameras > 0:
    cam = cameras[0][1]


    #get devcie serial number
    nodemap_tldevice = cam.GetTLDeviceNodeMap()


    cam.Init()

    # Retrieve GenICam nodemap
    nodemap = cam.GetNodeMap()

    node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
    if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
        device_serial_number = node_device_serial_number.GetValue()
        print('Device serial number retrieved as %s...' % device_serial_number)

    cam.BeginAcquisition()

    image_result = cam.GetNextImage(1000)

    # Getting the image data as a numpy array
    image_data = image_result.GetNDArray()

    image_result.Release()

def orderly_shutdown():
    cam.EndAcquisition()
    cam.DeInit()
    cam_list.Clear()
    system.ReleaseInstance()
