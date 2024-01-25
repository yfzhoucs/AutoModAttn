import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import numpy as np
from skimage import io, transform
from matplotlib import pyplot as plt
import json
import bisect
import clip
import random
from PIL import Image


class DMPDatasetEERandTarXYLang(Dataset):
    def __init__(self, data_dirs, normalize='none', length_total=91, image_size=(224, 224)):
        # |--datadir
        #     |--trial0
        #         |--img0
        #         |--img1
        #         |--imgx
        #         |--states.json
        #     |--trial1
        #     |--...

        assert normalize in ['none']

        all_dirs = []
        for data_dir in data_dirs:
            all_dirs = all_dirs + [ f.path for f in os.scandir(data_dir) if f.is_dir() ]
        # print(all_dirs)
        # print(len(all_dirs))
        self.random = random
        self.normalize = normalize
        self.length_total = length_total
        self.image_size = image_size
        self.trials = []
        self.lengths_index = []


        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])

        length = 0
        for trial in all_dirs:

            # trial_id = int(trial.strip().split(r'/')[-1])
            # if not ((trial_id >= 1700) and (trial_id < 1725)):
            #     continue


            trial_dict = {}

            states_json = os.path.join(trial, 'states_ee.json')
            with open(states_json) as json_file:
                states_dict = json.load(json_file)
                json_file.close()
            
            # There are (trial_dict['len']) states
            trial_dict['len'] = len(states_dict)
            trial_dict['img_paths'] = [os.path.join(trial, str(i) + '_left.jpg') for i in range(trial_dict['len'])]
            trial_dict['joint_angles'] = np.asarray([states_dict[i]['joints'] for i in range(trial_dict['len'])])
            trial_dict['gripper_position'] = np.asarray([[states_dict[i]['gripper_position'] / 255. for i in range(trial_dict['len'])]]).T
            trial_dict['joint_angles'] = np.concatenate((trial_dict['joint_angles'], trial_dict['gripper_position']), axis=1)
            
            trial_dict['EE_xyzrpy'] = np.asarray([states_dict[i]['objects_to_track']['EE']['xyz'] + self.rpy2rrppyy(states_dict[i]['objects_to_track']['EE']['rpy']) for i in range(trial_dict['len'])])
            
            # There are (trial_dict['len']) steps in the trial, which means (trial_dict['len'] + 1) states
            trial_dict['len'] -= 1
            self.trials.append(trial_dict)
            length = length + trial_dict['len']
            self.lengths_index.append(length)


    def rpy2rrppyy(self, rpy):
        rrppyy = [0] * 6
        for i in range(3):
            rrppyy[i * 2] = np.sin(rpy[i])
            rrppyy[i * 2 + 1] = np.cos(rpy[i])
        return rrppyy
    
    def bbox2center(self, bbox):
        return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2



    def __len__(self):
        return self.lengths_index[-1]

    def __getitem__(self, index):
        trial_idx = bisect.bisect_right(self.lengths_index, index)
        if trial_idx == 0:
            step_idx = index
        else:
            step_idx = index - self.lengths_index[trial_idx - 1]

        img = Image.open(self.trials[trial_idx]['img_paths'][step_idx])
        shape = img.size
        img = np.array(img.resize(self.image_size))[:,:,:3] / 255.
        img = img - self.imagenet_mean
        img = img / self.imagenet_std
        img = torch.tensor(img, dtype=torch.float32)


        length = torch.tensor(self.trials[trial_idx]['len'] - step_idx, dtype=torch.float32)
        ee_pos = torch.tensor((self.trials[trial_idx]['EE_xyzrpy'][step_idx]), dtype=torch.float32)
        ee_traj = torch.tensor((self.trials[trial_idx]['EE_xyzrpy'][step_idx:]), dtype=torch.float32)


        sentence = clip.tokenize(['pick duck'])


        joint_angles = torch.tensor(self.trials[trial_idx]['joint_angles'][step_idx], dtype=torch.float32)
        joint_angles_traj = torch.tensor(self.trials[trial_idx]['joint_angles'][step_idx:], dtype=torch.float32)

        length_total = self.length_total
        length_left = max(length_total - ee_traj.shape[0], 0)

        if length_left > 0:
            ee_traj_appendix = ee_traj[-1:].repeat(length_left, 1)
            ee_traj = torch.cat((ee_traj, ee_traj_appendix), axis=0)

            joint_angles_traj_appendix = joint_angles_traj[-1:].repeat(length_left, 1)
            joint_angles_traj = torch.cat((joint_angles_traj, joint_angles_traj_appendix), axis=0)

            # displacement_traj_appendix = displacement_traj[-1:].repeat(length_left, 1)
            # displacement_traj = torch.cat((displacement_traj, displacement_traj_appendix), axis=0)
        else:
            ee_traj = ee_traj[:length_total]
            joint_angles_traj = joint_angles_traj[:length_total]
            # displacement_traj = displacement_traj[:length_total]

        return img, joint_angles, ee_pos, ee_traj, length, sentence[0], joint_angles_traj


def pad_collate_xy_lang(batch):
    (img, joint_angles, ee_pos, ee_traj, length, sentence, joint_angles_traj) = zip(*batch)

    img = torch.stack(img)
    joint_angles = torch.stack(joint_angles)
    ee_pos = torch.stack(ee_pos)
    length = torch.stack(length)
    ee_traj = torch.nn.utils.rnn.pad_sequence(ee_traj, batch_first=True, padding_value=0)
    ee_traj = torch.transpose(ee_traj, 1, 2)
    sentence = torch.stack(sentence)
    joint_angles_traj = torch.nn.utils.rnn.pad_sequence(joint_angles_traj, batch_first=True, padding_value=0)
    joint_angles_traj = torch.transpose(joint_angles_traj, 1, 2)

    return img, joint_angles, ee_pos, ee_traj, length, sentence, joint_angles_traj


if __name__ == '__main__':
    data_dirs = [
        '/data2/yzhou298/dataset/pick_duck/pick_duck_1',
    ]
    dataset = DMPDatasetEERandTarXYLang(data_dirs, length_total=24)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2,
                                          shuffle=False, num_workers=1,
                                          collate_fn=pad_collate_xy_lang)

    for img, joint_angles, ee_pos, ee_traj, length, sentence, joint_angles_traj in dataloader:
        # print(target, joint_angles, ee_pos, ee_traj, length, target_pos)
        # print(length, len(ee_traj))
        print(img.shape, joint_angles.shape, ee_pos.shape, ee_traj.shape, length.shape, sentence.shape, joint_angles_traj.shape)

        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(Image.fromarray(((img[0].numpy() * dataset.imagenet_std + dataset.imagenet_mean) * 255).astype('uint8'), 'RGB'))

        ax = fig.add_subplot(1, 2, 2)
        xs = np.arange(joint_angles_traj.shape[2])
        ax.plot(xs, joint_angles_traj[0, 0, :], label='0')
        ax.plot(xs, joint_angles_traj[0, 1, :], label='1')
        ax.plot(xs, joint_angles_traj[0, 2, :], label='2')
        ax.plot(xs, joint_angles_traj[0, 3, :], label='3')
        ax.plot(xs, joint_angles_traj[0, 4, :], label='4')
        ax.plot(xs, joint_angles_traj[0, 5, :], label='5')
        ax.plot(xs, joint_angles_traj[0, 6, :], label='6')
        ax.legend()
        
        plt.show()
