#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import h5py
import os
import numpy as np
from tqdm import tqdm
import argparse

def save_events_h5(events, event_file):
    ex = events[:, 0].astype(np.uint16)
    ey = events[:, 1].astype(np.uint16)
    et = events[:, 2].astype(np.float32)
    ep = events[:, -1].astype(np.int8)

    file = h5py.File(event_file, 'w')
    file.create_dataset('x', data=ex, dtype=np.uint16, compression="lzf")
    file.create_dataset('y', data=ey, dtype=np.uint16, compression="lzf")
    file.create_dataset('p', data=ep, dtype=np.int8, compression="lzf")
    file.create_dataset('t', data=et, dtype=np.float32, compression="lzf")
    file.close()

def read_event_h5(path, chunk_indices):
    file = h5py.File(path, 'r')
    events = np.float32(file["events"][chunk_indices[0][0]:chunk_indices[-1][1]])[:,[1,2,0,3]]
    file.close()
    return events

def isskip(file, number_of_files):
    path = os.path.sep.join(file.split("/")[:-1]) 
    path = os.path.join(path,"event_left")
    
    if len(glob.glob(os.path.join(path,"*.hdf5"))) == number_of_files:
        return True 

    return False
    

def save_events(events, file, chunk_number):
    path = os.path.sep.join(file.split("/")[:-1]) 
    path = os.path.join(path,"event_left")
    
    os.makedirs(path,exist_ok=True)
    
    save_events_h5(events, os.path.join(path,f"{str(chunk_number).zfill(6)}_{str(chunk_number+1).zfill(6)}_event.hdf5"))
        

def save_chunks(path, n_image_files, chunk_size=10**6):
    # Read file, get end time and total number of events
    file = h5py.File(path, 'r')
    end_time = np.float32(file["events"][-1])[0]
    tot_events = len(file["events"])
    chunk_n = 0

    if np.round(end_time/10**6) != (n_image_files-1):
        print(f"Error: mismatch between end time {end_time/10**6} and number of image files {(n_image_files-1)}")
        exit()

    # Define the number of events to read at once
    read_size = int((tot_events/n_image_files)*250)

    # Create an array of time values that we want to find in the time array
    value = []
    for i in range(0,n_image_files+1):
        value.append(i*10**6)

    for i in range(0, tot_events, read_size):
        # Read a part of the events
        events = np.float32(file["events"][i:i+read_size])[:,[1,2,0,3]]
        times = events[:, 2]

        # Find the indices of the time values in the time array
        start_idx = int(times[0]/chunk_size)
        end_idx = int(times[-1]/chunk_size)
        value_sec = value[start_idx:end_idx+1]
        idx = np.searchsorted(times, value_sec)

        # Save the chunks, based on the indices
        for id in range(len(idx)-1):
            # print(f"Processing chunk {chunk_n}")
            if start_idx != 0 and id == 0:
                cat_events = np.concatenate((last_events, events[0:idx[id+1]]), axis=0)
                save_events(cat_events, path, chunk_n)
                chunk_n += 1
            else:
                save_events(events[idx[id]:idx[id+1]], path, chunk_n)
                chunk_n += 1

            if id == (len(idx)-2):
                # save the last events, to be concatenated with the first events of the next chunk
                last_events = events[idx[id+1]:]

    save_events(events[idx[-1]:], path, chunk_n)

def valid_file(file):
    image_files = file.split("/")[:-1] + ["image_left", "*.png"]
    image_files = len(glob.glob(os.path.sep.join(image_files))) - 1
    
    # skip if the file is already processed
    if isskip(file, image_files):
        print(f"Skip {file}")
        return 
    
    # variables
    chunk_size = 10**6      # time bins

    print(f"Processing {file}")
    save_chunks(file, image_files, chunk_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_template', default="./dataset/**/event_left.h5", type=str, required=True)
    option = parser.parse_args()
    for file in tqdm(glob.glob(option.file_template,recursive=True)):
        valid_file(file)