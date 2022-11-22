from typing import OrderedDict
from dv import AedatFile
import numpy as np
import tarfile
import os.path
import cv2
from PIL import Image
def aedat4_to_npz(file, target="/home/guy/Projects/Results/Davis/npz/mug.npz"):
    with AedatFile(file) as f:
        # events will be a named numpy array
        events = np.hstack([packet for packet in f['events'].numpy()])

        # Access information of all events by type
        timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']

        npz = OrderedDict()
        npz['t'] = timestamps
        npz['x'] = x
        npz['y'] = y
        npz['p'] = polarities
        
        np.savez(target, **npz)

def aedat4_to_txt(file):
    with AedatFile(file) as f:
        info = f['events']._stream_info
        txt_file = file.replace(".aedat4", ".txt")
        events = np.hstack([packet for packet in f['events'].numpy()])
        
        num_events = len(events)
        d_time = (events[len(events)-1][0] - events[0][0]) /10e5
        num_pixels = int(info['sizeX'])*int(info['sizeY'])
        print("events/pixel,second = " + str(num_events/(num_pixels*d_time)))

        with open(txt_file, 'w') as t:
            t.write(info['sizeX']+" "+info['sizeY']+"\n")
            for event in events:
                t.write(str(event[0]/10e5)+" "+str(event[1])+" "+str(event[2])+" "+str(event[3])+"\n")

# get frames from aedat4 file
def getImages(file, target_folder="/home/guy/Projects/Results/Davis/gt_images"):
    with AedatFile(file) as f:
        frames =  f['frames']
        for frame in frames:
            cv2.imwrite(os.path.join(target_folder, str(frame.timestamp))+'.png', frame.image)



if __name__ == "__main__":
    file = "/home/guy/Projects/Results/Davis/mug.aedat4"
    # aedat4_to_txt(file)
    # aedat4_to_npz(file)
    getImages(file)