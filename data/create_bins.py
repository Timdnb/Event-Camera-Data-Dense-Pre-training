import os
os.environ["KMP_BLOCKTIME"] = "0"
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import torch
from event_utils import eventsToVoxel
from file_io import read_event_h5

def store_voxel(event, event_bins, event_polarity, height, width, voxel_path, event_file):
    if event.shape[0] < 10:
        c = 1 + int(event_polarity)
        event  = np.zeros((event_bins*c,height,width))
    else:
        event = eventsToVoxel(event, num_bins=event_bins, height=height, width=width, event_polarity=event_polarity, temporal_bilinear=True)
            
    event = torch.from_numpy(event)

    event_file = event_file[:-5]
    voxel_file = os.path.join(voxel_path, event_file + ".pt")

    torch.save(event, voxel_file)

if __name__ == "__main__":
    event_bins = 6
    event_polarity = False
    height, width = 480, 640

    dataset_path = "/data/tim/datasets/E-TartanAir"

    for env in os.listdir(dataset_path):
        env_path = os.path.join(dataset_path, env)
        if os.path.isfile(env_path):
            continue
        for diff in os.listdir(env_path):
            if diff == "Hard":
                continue
            diff_path = os.path.join(env_path, diff)
            for scene in os.listdir(diff_path):
                scene_path = os.path.join(diff_path, scene)
                voxel_path = os.path.join(scene_path, f"voxels_{event_bins}")
                if not os.path.exists(voxel_path):
                    os.mkdir(voxel_path)
                elif os.path.exists(os.path.join(scene_path, f"voxels_{event_bins}")) and len(os.listdir(os.path.join(scene_path, f"voxels_{event_bins}"))) == len(os.listdir(os.path.join(scene_path, f"event_left"))):
                    print(f"Skipping {env}/{diff}/{scene}: voxels already created")
                    continue
                for event in os.listdir(os.path.join(scene_path, "event_left")):
                    event_folder = os.path.join(scene_path, "event_left")
                    event_path = os.path.join(event_folder, event)
                    events = read_event_h5(event_path)
                    store_voxel(events, event_bins, event_polarity, height, width, voxel_path, event)
            
                print(f"Finished creating voxels for {env}/{diff}/{scene}")