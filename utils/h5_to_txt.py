import numpy as np
from collections import  OrderedDict
import os
import h5py
import argparse
from tqdm import tqdm
from zipfile import ZipFile
import pathlib

def conver_h5_folder_to_txt(src="/home/guy/Projects/Results/speed/h5_data", dest="/home/guy/Projects/Results/speed/txt_data", zip=True, delete=True):
    pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
    h5_files = os.listdir(src)
    for h5 in tqdm(h5_files):
        h5_path = os.path.join(src, h5)
        txt = h5.replace('h5', "txt")
        txt_path = os.path.join(dest, txt)
        h5_to_txt(h5_file_path=h5_path, txt_file_path=txt_path)
    
    if(zip):
        delete_txt = "and deleting" if delete else ""
        print(f"zipping {delete_txt} txt files")
        for txt in tqdm(os.listdir(dest)):
            txt_path = os.path.join(dest, txt)
            zip_path = txt_path.replace('.txt', '.zip')
            with ZipFile(zip_path,'w') as zip:
                zip.write(txt_path)
            if(delete):
                os.remove(txt_path)
            
def h5_to_txt(h5_file_path, txt_file_path):
    resolution, events = read_h5_events(h5_file_path)
    with open(txt_file_path, 'w') as t:
        t.write(str(resolution[1])+" "+str(resolution[0])+"\n")
        for event in events:
            t.write(str(event[0])+" "+str(event[1])+" "+str(event[2])+" "+str(event[3])+"\n")
            
def read_h5_events(hdf_path):
   
    with h5py.File( hdf_path, 'r') as f:
    
        ts = f['events/ts'][:]
        xs = f['events/xs'][:]
        ys = f['events/ys'][:]
        ps = np.where(f['events/ps'][:], 1, -1)
        events = [packet for packet in zip(ts, xs, ys, ps )]

        return  f.attrs['sensor_resolution'], events


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convertion of all npz files in a folder to a single h5')
    parser.add_argument('-s', '--source_folder', required=True, type=str,
                        help='path to the parent folder containing the h5 files')
    parser.add_argument('-o', '--output_folder', required=True, type=str)
    parser.add_argument('-z', '--zip', default=True, type=bool, help="seperately zip all text files")
    parser.add_argument('-d', '--delete', default=True, type=bool, help="delete txt file - only if zipped first")
    args = parser.parse_args()
    conver_h5_folder_to_txt(args.source_folder, args.output_folder, args.zip, args.delete)