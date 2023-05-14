import cv2
import numpy as np
import argparse
import os
import sys
import random
from utils import *



def create_imgfilepath_txt():


    root_dir = 'D:/hongRui/GMU_course/CS782/Project/src/data/CDnet2014'
    root_data_dir = 'D:/hongRui/GMU_course/CS782/Project/src/data/CDnet2014/data'

    

    train_filepaths = []
    test_filepaths = []
    # train_background_filepath = []
    # test_background_filepath = []
    background_filepaths = []
    for i, first_name in enumerate(os.listdir(root_data_dir)):
        first_abs_dir = os.path.join(root_data_dir, first_name)#, second_dir, third_dir)
        # for j, second_name in enumerate(os.listdir(first_abs_dir)):
        #     second_abs_dir = os.path.join(first_abs_dir, second_name)#, second_dir, third_dir)
        for k, third_name in enumerate(os.listdir(first_abs_dir)):
            third_abs_dir = os.path.join(first_abs_dir, third_name)
            temporalROI_filepath = os.path.join(third_abs_dir, 'temporalROI.txt') 
            min_n, max_n = get_temporalROI_from_txt(temporalROI_filepath)
            img_dir = os.path.join(third_abs_dir, 'input') 
            lbl_dir = os.path.join(third_abs_dir, 'groundtruth') 
            first_flag = True
            positive_img_filepaths = []
            for num in range(min_n, max_n):
                img_name = 'in' + str(num).rjust(6, '0')
                lbl_name = 'gt' + str(num).rjust(6, '0')
                img_filepath = os.path.join(img_dir, img_name  + ".jpg")
                lbl_filepath = os.path.join(lbl_dir, lbl_name  + ".png")
                print(f'{i}/{len(os.listdir(root_data_dir))}, {k}/{len(os.listdir(third_abs_dir))}, {num}/{max_n}, {img_filepath}')

                if os.path.exists(img_filepath) and os.path.exists(lbl_filepath):
                    label = cv2.imread(lbl_filepath, 0)
                    mask_bk = label.copy()
                    if len(label[mask_bk != 170]) == 0 or len(label[mask_bk != 85]) == 0:
                        continue
                                          
                    label[mask_bk == 0] = 0
                    label[mask_bk == 50] = 1                        
                    label[mask_bk == 255] = 1
                    label[mask_bk == 170] = 0
                    label[mask_bk==85]=0
                    if len(label[label>0]) <= 0:
                        if first_flag:      
                            background_filepaths.append(img_filepath)
                            first_flag = False
                    else:
                        positive_img_filepaths.append(img_filepath)

            random.shuffle(positive_img_filepaths)
            for m, img_filepath in enumerate(positive_img_filepaths):
                if m < len(positive_img_filepaths)/2:
                    train_filepaths.append(img_filepath)
                else:
                    test_filepaths.append(img_filepath)
                    
    txt_filepath = os.path.join(root_dir, 'train.txt')
    if os.path.exists(txt_filepath):
        os.remove(txt_filepath)
    write_list_to_txt(txt_filepath, train_filepaths)

    txt_filepath = os.path.join(root_dir, 'test.txt')
    if os.path.exists(txt_filepath):
        os.remove(txt_filepath)
    write_list_to_txt(txt_filepath, test_filepaths)
        
    
    txt_filepath = os.path.join(root_dir, 'background.txt')
    if os.path.exists(txt_filepath):
        os.remove(txt_filepath)
    write_list_to_txt(txt_filepath, background_filepaths)

    # txt_filepath = os.path.join(root_dir, 'test_background.txt')
    # if os.path.exists(txt_filepath):
    #     os.remove(txt_filepath)
    # write_list_to_txt(txt_filepath, test_background_filepath)

def process_background_filepath():
    root_dir = 'D:/hongRui/GMU_course/CS782/Project/src/data/CDnet2014'
    txt_filepath = os.path.join(root_dir, 'background.txt')

    background_filepaths = read_txt_to_list(txt_filepath)
    for img_path in background_filepaths:
        img_path = img_path.replace('\\', '/')
            
        img_name = img_path.split('/')[-1]
        img_dir = img_path[:img_path.find(img_name)]


        # lbl_path = img_path.replace('image', 'label').replace('.jpg', '.png')
        bk_filepath = os.path.join(img_dir[:img_dir.find('input')], 'background.jpg')
        shutil.copy(img_path, bk_filepath)



def process_bmc_data(data_index, split):
    img_video_dir = f'D:/hongRui/GMU_course/CS782/Project/src/data/bmc_synth{data_index}/input'
    gt_video_dir = f'D:/hongRui/GMU_course/CS782/Project/src/data/bmc_synth{data_index}/gt'
    output_dir = f'D:/hongRui/GMU_course/CS782/Project/src/data/bmc/{split}'

    video_names = os.listdir(img_video_dir)
    
    for i, v_name in enumerate(video_names):
        img_video_filepath = os.path.join(img_video_dir, v_name)
        gt_name = v_name.split('.')[0] + '_gt.mp4'
        gt_video_filepath = os.path.join(gt_video_dir, gt_name)
        img_cam = cv2.VideoCapture(img_video_filepath)
        gt_cam = cv2.VideoCapture(gt_video_filepath)

        total_img_num = int(img_cam.get(cv2.CAP_PROP_FRAME_COUNT))
        total_gt_num = int(gt_cam.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(f'{v_name}, total_img_num: {total_img_num}, total_gt_num: {total_gt_num}')
        output_img_dir = os.path.join(output_dir, v_name.split('.')[0], 'image')
        output_gt_dir = os.path.join(output_dir, v_name.split('.')[0], 'groundtruth')
        os.makedirs(output_img_dir,exist_ok=True)
        os.makedirs(output_gt_dir,exist_ok=True)
        # frame
        currentframe = 0
        
        ret, gt = gt_cam.read()
        print(f'gt.shape: {gt.shape}')
        while(True):
            
            # reading from frame
            img_ret, img = img_cam.read()
            gt_ret, gt = gt_cam.read()
            # print(f'{currentframe}, gt.shape: {gt.shape}')
            print(f'{currentframe}, {v_name}, {i}/ {len(video_names)}')
            
            currentframe += 1
            if not currentframe % 3 == 0:
                continue
            if img_ret and gt_ret:
                gt = gt[:,:,0]
                img_name = v_name.split('.')[0] + f"_{str(currentframe).rjust(5, '0')}.jpg"
                gt_name = v_name.split('.')[0] + f"_{str(currentframe).rjust(5, '0')}.png"
                img_filepath = os.path.join(output_img_dir, img_name)
                gt_filepath = os.path.join(output_gt_dir, gt_name)
                cv2.imwrite(img_filepath, img)
                cv2.imwrite(gt_filepath, gt)
        
                # increasing counter so that it will
                # show how many frames are created
                
            else:
                break
        
        # Release all space and windows once done
        img_cam.release()
        gt_cam.release()
        cv2.destroyAllWindows()



def create_imgfilepath_txt_for_bmc(split):


    root_dir =      'D:/hongRui/GMU_course/CS782/Project/src/data/bmc/'
    root_data_dir = f'D:/hongRui/GMU_course/CS782/Project/src/data/bmc/{split}'

    

    # train_filepaths = []
    # test_filepaths = []
    # train_background_filepath = []
    # test_background_filepath = []
    positive_img_filepaths = []
    for i, first_name in enumerate(os.listdir(root_data_dir)):
        first_abs_dir = os.path.join(root_data_dir, first_name)#, second_dir, third_dir)
        img_dir = os.path.join(first_abs_dir, 'image') 
        lbl_dir = os.path.join(first_abs_dir, 'groundtruth') 
        img_names = os.listdir(img_dir)
        first_flag = True
        
        for j, img_name in enumerate(img_names):
            img_filepath = os.path.join(img_dir, img_name)
            lbl_filepath = img_filepath.replace('image', 'groundtruth').replace('.jpg', '.png')
            print(f'{i}/{len(os.listdir(root_data_dir))},  {j}/{len(img_names)} {img_filepath}')

            if os.path.exists(img_filepath) and os.path.exists(lbl_filepath):
                label = cv2.imread(lbl_filepath, 0)
                
                if len(label[label>0]) <= 0:
                    if first_flag:      
                        bk_filepath = os.path.join(first_abs_dir, 'background.jpg')
                        shutil.copy(img_filepath, bk_filepath)
                        first_flag = False
                else:
                    positive_img_filepaths.append(img_filepath)

                
    txt_filepath = os.path.join(root_dir, f'{split}.txt')
    if os.path.exists(txt_filepath):
        os.remove(txt_filepath)
    write_list_to_txt(txt_filepath, positive_img_filepaths)

    # txt_filepath = os.path.join(root_dir, 'test.txt')
    # if os.path.exists(txt_filepath):
    #     os.remove(txt_filepath)
    # write_list_to_txt(txt_filepath, test_filepaths)
        
    


# create_imgfilepath_txt()
# process_background_filepath()
process_bmc_data(1, 'train')
create_imgfilepath_txt_for_bmc('train')

process_bmc_data(2, 'test')
create_imgfilepath_txt_for_bmc('test')