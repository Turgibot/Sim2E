import numpy as np
import pandas as pd
import h5py
import argparse
from tqdm import tqdm
import csv
import os
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity
import cv2
import lpips

def create_csv(src="h5_data", dest="results.csv"):
    file = os.path.join(src, os.listdir(src)[0])
    num = file.split('/')
    num = num[len(num)-1].replace('.h5', '')
    header = ['number']

    with h5py.File(file, 'r') as f:
        for k in f.attrs.keys():
            if k=="sensor_resolution":
                header.append("height")
                header.append("width")
                continue
            header.append(k)
        loss = ["ours_ssim","ours_lpips","e2v_ssim","e2v_lpips","fire_ssim","fire_lpips","ecm_ssim","ecm_lpips"]
        [header.append(l) for l in loss]

        with open(dest, mode='w') as csv_file:
            
            writer = csv.writer(csv_file)
            writer.writerow(header)
            # meta-data
            files = os.listdir(src)
            for file in tqdm(files):
                num = file.split('/')
                num = num[len(num)-1].replace('.h5', '')
                data = [int(num)]
                h5_file = os.path.join(src, file)
                with h5py.File(h5_file, 'r') as f:
                    for k in f.attrs.keys():
                        if k=="sensor_resolution":
                            res = f.attrs[k]
                            data.append(res[0])
                            data.append(res[1])
                            continue
                        data.append(f.attrs[k])
                    writer.writerow(data)

def add_losses_to_csv(csv_path, gt_folder, inference_folder):
    
    df = pd.read_csv(csv_path)
    df = df.sort_values("number")
    df.reset_index(inplace=True, drop=True)
    # print(df.to_string()) 
    loss_cols = df.columns.values.tolist()[-8:]
    scenes = os.listdir(gt_folder)
    scenes.sort()
    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    transform = transforms.ToTensor()

    for scene in scenes:
        e2vid_inferences = os.path.join(inference_folder, "e2vid", scene)
        ecm_inferences = os.path.join(inference_folder, "ecm", scene)
        fire_inferences = os.path.join(inference_folder, "fire", scene)
        our_inferences = os.path.join(inference_folder, "ours", scene)
        
        scene_gt_folder_path = os.path.join(gt_folder, scene)
        gts_images = os.listdir(scene_gt_folder_path)
        gts_images.sort()
        gts_images.pop() # remove first image inference

        ours = os.listdir(our_inferences)
        ours.sort()
        ours.pop()
        ours.pop(0)

        event_cnn_min = os.listdir(ecm_inferences)
        event_cnn_min.sort()
        event_cnn_min.pop()
        event_cnn_min.pop(0)

        firenet = os.listdir(fire_inferences)
        firenet.sort()
        firenet.pop()
        
        e2vid = os.listdir(e2vid_inferences)
        e2vid.sort()
        e2vid.pop()

        gts_color = [cv2.imread(os.path.join(scene_gt_folder_path, img)) for img in gts_images]
        gts_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in gts_color ]

        ours_color = [cv2.imread(os.path.join(our_inferences, img)) for img in ours]
        ours_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in ours_color]

        event_cnn_mins_color = [cv2.imread(os.path.join(ecm_inferences, img)) for img in event_cnn_min]
        event_cnn_mins_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in event_cnn_mins_color]

        firenets_color = [cv2.imread(os.path.join(fire_inferences, img)) for img in firenet]
        firenets_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in firenets_color]

        e2vids_color = [cv2.imread(os.path.join(e2vid_inferences, img)) for img in e2vid]
        e2vids_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in e2vids_color]

        len_gt = len(gts_images)
        len_ours = len(ours)
        len_event_cnn = len(event_cnn_min)
        len_firenet = len(firenet)
        len_e2vid = len(e2vid)


        sum_ssim_ours, sum_ssim_event_cnn, sum_ssim_firenet, sum_ssim_e2vid = 0, 0, 0, 0
        sum_lpips_ours, sum_lpips_event_cnn, sum_lpips_firenet, sum_lpips_e2vid = 0, 0, 0, 0

        

        for i in tqdm(range(len_gt)):
            gt_gray = gts_gray[i] 
            gt_color = gts_color[i] 

            our_gray = ours_gray[int((i/len_gt)*len_ours)]
            our_color = ours_color[int((i/len_gt)*len_ours)]

            event_cnn_min_gray = event_cnn_mins_gray[int((i/len_gt)*len_event_cnn)]
            event_cnn_min_color = event_cnn_mins_color[int((i/len_gt)*len_event_cnn)]

            firenet_gray = firenets_gray[int((i/len_gt)*len_firenet)]
            firenet_color = firenets_color[int((i/len_gt)*len_firenet)]

            e2vid_gray = e2vids_gray[int((i/len_gt)*len_e2vid)]
            e2vid_color = e2vids_color[int((i/len_gt)*len_e2vid)]

            # SSIM
            score_ours, _ = structural_similarity(gt_gray, our_gray, full=True)
            score_event_cnn, _ = structural_similarity(gt_gray, event_cnn_min_gray, full=True)
            score_firenet, _ = structural_similarity(gt_gray, firenet_gray, full=True)
            score_e2vid, _ = structural_similarity(gt_gray, e2vid_gray, full=True)
            
            sum_ssim_ours += score_ours
            sum_ssim_event_cnn += score_event_cnn
            sum_ssim_firenet += score_firenet
            sum_ssim_e2vid += score_e2vid

            # LPIPS
            gt_pil = Image.fromarray(gt_color).convert('RGB')
            our_pil = Image.fromarray(our_color).convert('RGB')
            event_pil = Image.fromarray(event_cnn_min_color).convert('RGB')
            firenet_pil = Image.fromarray(firenet_color).convert('RGB')
            e2vid_pil = Image.fromarray(e2vid_color).convert('RGB')

            gt_pil = transform(gt_pil)
            our_pil = transform(our_pil)
            event_pil = transform(event_pil)
            firenet_pil = transform(firenet_pil)
            e2vid_pil = transform(e2vid_pil)


            sum_lpips_ours += loss_fn_alex(gt_pil, our_pil)
            sum_lpips_event_cnn += loss_fn_alex(gt_pil, event_pil)
            sum_lpips_firenet += loss_fn_alex(gt_pil, firenet_pil)
            sum_lpips_e2vid += loss_fn_alex(gt_pil, e2vid_pil)
        
        results = []
        results.append(sum_ssim_ours/len_gt)
        results.append((sum_lpips_ours/len_gt).item())
        results.append(sum_ssim_e2vid/len_e2vid)
        results.append((sum_lpips_e2vid/len_e2vid).item())
        results.append(sum_ssim_firenet/len_gt)
        results.append((sum_lpips_firenet/len_gt).item())
        results.append(sum_ssim_event_cnn/len_gt)
        results.append((sum_lpips_event_cnn/len_gt).item())
            
        
        row_index = df.index[df['number'] == int(scene)]
        for j, result in enumerate(results):
            df.loc[row_index, loss_cols[j]] = result
        print(row_index[0])
    df.to_csv(csv_path)
        


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='collect results from loss analysis to a single csv')
    # parser.add_argument('-s', '--source_folder', required=True, type=str,
    #                     help='path to the parent folder containing the h5 files')
    # parser.add_argument('-c', '--csv_file', required=True, type=str)
    # parser.add_argument('-i', '--inference_folder', default="inference", type=str)
    # parser.add_argument('-gt', '--gt_folder', default="gt_images", type=str)

    # args = parser.parse_args()
    # create_csv(args.source_folder, args.csv_file)
    # add_losses_to_csv(args.csv_file, args.gt_folder, args.inference_folder)
    add_losses_to_csv("/home/guy/Projects/Results/Davis/davis_results.csv", "/home/guy/Projects/Results/Davis/gt_images", "/home/guy/Projects/Results/Davis/inference")