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
    def __init__(self, data_dirs, random=True, normalize='none', length_total=91, image_size=(224, 224)):
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
        self.target_name_to_idx = {
            'cube': 0,
            'can': 1,
            'fanta': 2,
            'milk_carton': 3,
            'bottle': 4,
        }

        self.idx_to_name = {
            0: 'a red cube',
            1: 'a pepsi can',
            2: 'a fanta bottle',
            3: 'a milk carton',
            4: 'a sprite bottle',
        }

        self.action_inst_to_verb = {
            'push': ['push', 'move'],
            'pick': ['pick', 'pick up', 'raise', 'hold'],
            'rotate': ['rotate', 'turn down']
        }

        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])

        length = 0
        for trial in all_dirs:

            # trial_id = int(trial.strip().split(r'/')[-1])
            # if not ((trial_id >= 1700) and (trial_id < 1725)):
            #     continue


            trial_dict = {}

            states_json = os.path.join(trial, 'states_ee_bboxes.json')
            with open(states_json) as json_file:
                states_dict = json.load(json_file)
                json_file.close()
            
            # There are (trial_dict['len']) states
            trial_dict['len'] = len(states_dict)
            trial_dict['img_paths'] = [os.path.join(trial, str(i) + '.jpg') for i in range(trial_dict['len'])]
            trial_dict['joint_angles'] = np.asarray([states_dict[i]['joints'] for i in range(trial_dict['len'])])
            trial_dict['gripper_position'] = np.asarray([[states_dict[i]['gripper_position'] / 255. for i in range(trial_dict['len'])]]).T
            trial_dict['joint_angles'] = np.concatenate((trial_dict['joint_angles'], trial_dict['gripper_position']), axis=1)
            
            trial_dict['EE_xyzrpy'] = np.asarray([states_dict[i]['objects_to_track']['EE']['xyz'] + self.rpy2rrppyy(states_dict[i]['objects_to_track']['EE']['rpy']) for i in range(trial_dict['len'])])
            
            # trial_dict['a red cube'] = np.asarray([self.bbox2center(states_dict[i]['bboxes']['a red cube']) for i in range(trial_dict['len'])])
            # trial_dict['a pepsi can'] = np.asarray([self.bbox2center(states_dict[i]['bboxes']['a pepsi can']) for i in range(trial_dict['len'])])
            # trial_dict['a fanta bottle'] = np.asarray([self.bbox2center(states_dict[i]['bboxes']['a fanta bottle']) for i in range(trial_dict['len'])])
            # trial_dict['a milk carton'] = np.asarray([self.bbox2center(states_dict[i]['bboxes']['a milk carton']) for i in range(trial_dict['len'])])
            # trial_dict['a sprite bottle'] = np.asarray([self.bbox2center(states_dict[i]['bboxes']['a sprite bottle']) for i in range(trial_dict['len'])])
            
                        
            trial_dict['a red cube'] = self.get_seq_bboxes(states_dict, 'a red cube')
            trial_dict['a pepsi can'] = self.get_seq_bboxes(states_dict, 'a pepsi can')
            trial_dict['a fanta bottle'] = self.get_seq_bboxes(states_dict, 'a fanta bottle')
            trial_dict['a milk carton'] = self.get_seq_bboxes(states_dict, 'a milk carton')
            trial_dict['a sprite bottle'] = self.get_seq_bboxes(states_dict, 'a sprite bottle')
            

            trial_dict['target_id'] = states_dict[0]['tar_obj']
            trial_dict['action_inst'] = states_dict[0]['action']
            
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
    
    def get_seq_bboxes(self, states_dict, object_name):
        bboxes = []
        masks = []
        length = len(states_dict)
        for i in range(length):
            if object_name in states_dict[i]["bboxes"]:
                bboxes.append(np.array(self.bbox2center(states_dict[i]["bboxes"][object_name])))
                masks.append(np.array([1, 1]))
            else:
                bboxes.append(np.array([0, 0]))
                masks.append(np.array([0, 0]))
        return {'center': bboxes, 'mask': masks}


    def noun_phrase_template(self, target_id):
        self.noun_phrase = {
            0: {
                'name': ['red', 'maroon'],
                'object': ['object', 'cube', 'square'],
            },
            1: {
                'name': ['red', 'coke', 'cocacola'],
                'object': ['can', 'bottle'],
            },
            2: {
                'name': ['blue', 'pepsi', 'pepsi coke'],
                'object': ['can', 'bottle'],
            },
            3: {
                'name': ['milk', 'white'],
                'object': ['carton', 'box'],
            },
            4: {
                'name': ['bread', 'yellow object', 'brown object'],
                'object': [''],
            },
            5: {
                'name': ['green', '', 'glass', 'green glass'],
                'object': ['bottle'],
            }
        }
        id_name = np.random.randint(len(self.noun_phrase[target_id]['name']))
        id_object = np.random.randint(len(self.noun_phrase[target_id]['object']))
        name = self.noun_phrase[target_id]['name'][id_name]
        obj = self.noun_phrase[target_id]['object'][id_object]
        return (name + ' ' + obj).strip()

    def verb_phrase_template(self, action_inst):
        if action_inst is None:
            action_inst = random.choice(list(self.action_inst_to_verb.keys()))
        action_id = np.random.randint(len(self.action_inst_to_verb[action_inst]))
        verb = self.action_inst_to_verb[action_inst][action_id]
        return verb.strip()

    def sentence_template(self, target_id, action_inst=None):
        sentence = ''
        verb = self.verb_phrase_template(action_inst)
        sentence = sentence + verb
        sentence = sentence + ' ' + self.noun_phrase_template(target_id)
        return sentence.strip()

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


        if self.random:
            target = np.random.randint(5)
            action = None
        else:
            target = self.target_name_to_idx[self.trials[trial_idx]['target_id']]
            action = self.trials[trial_idx]['action_inst']

        sentence = self.sentence_template(target, action)
        sentence = clip.tokenize([sentence])

        target_xy_center = self.trials[trial_idx][self.idx_to_name[target]]['center'][step_idx] * np.array(self.image_size) * np.array([1 / shape[0], 1 / shape[1]]) / 100.
        target_xy_mask = self.trials[trial_idx][self.idx_to_name[target]]['mask'][step_idx]
        target = torch.tensor(target, dtype=torch.int64)

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

        phis = torch.tensor(np.linspace(0.0, 1.0, length_total, dtype=np.float32))
        mask = torch.ones(phis.shape)

        return img, target, joint_angles, ee_pos, ee_traj, length, phis, mask, target_xy_center, target_xy_mask, sentence[0], joint_angles_traj


def pad_collate_xy_lang(batch):
    (img, target, joint_angles, ee_pos, ee_traj, length, phis, mask, target_xy_center, target_xy_mask, sentence, joint_angles_traj) = zip(*batch)

    img = torch.stack(img)
    target = torch.stack(target)
    joint_angles = torch.stack(joint_angles)
    ee_pos = torch.stack(ee_pos)
    length = torch.stack(length)
    ee_traj = torch.nn.utils.rnn.pad_sequence(ee_traj, batch_first=True, padding_value=0)
    ee_traj = torch.transpose(ee_traj, 1, 2)
    phis = torch.nn.utils.rnn.pad_sequence(phis, batch_first=True, padding_value=0).unsqueeze(1).repeat(1, ee_traj.shape[1] + 1, 1)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0).unsqueeze(1).repeat(1, ee_traj.shape[1] + 1, 1)
    target_xy_center = np.stack(target_xy_center)
    target_xy_mask = np.stack(target_xy_mask)
    sentence = torch.stack(sentence)
    joint_angles_traj = torch.nn.utils.rnn.pad_sequence(joint_angles_traj, batch_first=True, padding_value=0)
    joint_angles_traj = torch.transpose(joint_angles_traj, 1, 2)

    return img, target, joint_angles, ee_pos, ee_traj, length, phis, mask, target_xy_center, target_xy_mask, sentence, joint_angles_traj


if __name__ == '__main__':
    data_dirs = [
        '/data2/yzhou298/dataset/auto_modattn/ur5_table_top_manip/pick/',
        '/data2/yzhou298/dataset/auto_modattn/ur5_table_top_manip/pick_0/',
    ]
    dataset = DMPDatasetEERandTarXYLang(data_dirs, random=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2,
                                          shuffle=False, num_workers=1,
                                          collate_fn=pad_collate_xy_lang)

    for img, target, joint_angles, ee_pos, ee_traj, length, phis, mask, target_xy_center, target_xy_mask, sentence, joint_angles_traj in dataloader:
        # print(target, joint_angles, ee_pos, ee_traj, length, target_pos)
        # print(length, len(ee_traj))
        print(img.shape, target.shape, joint_angles.shape, ee_pos.shape, ee_traj.shape, length.shape, phis.shape, mask.shape, target_xy_center.shape, target_xy_mask.shape, sentence.shape, joint_angles_traj.shape)

        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(Image.fromarray(((img[0].numpy() * dataset.imagenet_std + dataset.imagenet_mean) * 255).astype('uint8'), 'RGB'))
        circle1 = plt.Circle(target_xy_center[0], 3, color='r')
        ax.add_patch(circle1)

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
