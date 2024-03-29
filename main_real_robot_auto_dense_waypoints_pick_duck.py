import numpy as np
# np.set_printoptions(precision=3, suppress=True)
from models.backbone_rgb_auto_dense_waypoints import Backbone
from utils.load_data_rgb_abs_action_auto_pick_duck import DMPDatasetEERandTarXYLang, pad_collate_xy_lang
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torch
from torch.optim.lr_scheduler import StepLR
import os
import matplotlib.pyplot as plt
import time
import random
import clip
import sys


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def pixel_position_to_attn_index(pixel_position, attn_map_offset=1):
    index = (pixel_position[:, 0]) // 8 + attn_map_offset + 28 * (pixel_position[:, 1] // 8)
    index = index.astype(int)
    index = torch.tensor(index).to(device).unsqueeze(1)
    return index


def attn_loss(attn_map, supervision, criterion, scale):
    # supervision = [[0, [-1]], [1, [1]], [2, [2]], [3, [3]], [4, [4]]]
    # supervision = [[1, [0, 2, 3]], [2, [2]], [4, [4]]]
    loss = 0
    for supervision_pair in supervision:
        target_attn = 0
        for i in supervision_pair[1]:
            target_attn = target_attn + attn_map[:, supervision_pair[0], i]
        loss = loss + criterion(target_attn, torch.ones(attn_map.shape[0], 1, dtype=torch.float32).to(device))
    loss = loss * scale
    return loss



def train(writer, name, epoch_idx, data_loader, model, 
    optimizer, scheduler, criterion, ckpt_path, save_ckpt, stage,
    print_attention_map=False, curriculum_learning=False, supervised_attn=False):
    model.train()
    criterion2 = nn.L1Loss(reduction='none')

    for idx, (img, joint_angles, ee_pos, ee_traj, length, sentence, joint_angles_traj) in enumerate(data_loader):
        global_step = epoch_idx * len(data_loader) + idx

        # Prepare data
        img = img.to(device)
        joint_angles = joint_angles.to(device)
        ee_pos = ee_pos.to(device)
        ee_traj = ee_traj.to(device)
        length = length.to(device)
        sentence = sentence.to(device)
        joint_angles_traj = joint_angles_traj.to(device)
        ee_traj = torch.cat((ee_traj, joint_angles_traj[:, -1:, :]), axis=1)



        # Forward pass
        optimizer.zero_grad()
        if stage == 0:
            attn_map, attn_map2 = model(img, joint_angles, sentence, None, stage)
        elif stage == 1:
            target_xy_pred, ee_pos_pred, attn_map, attn_map2, attn_map3 = model(img, joint_angles, sentence, None, stage)
        else:
            target_xy_pred, ee_pos_pred, attn_map, attn_map2, attn_map3, attn_map4, trajectory_pred = model(img, joint_angles, sentence, None, stage)


        # Attention Supervision for layer1
        supervision_layer1 = [[0, [-1]], [1, [1]], [2, [2]], [3, [3]], [4, [4]]]
        loss_attn_layer1 = attn_loss(attn_map, supervision_layer1, criterion, scale=5000)

        # Attention Supervision for layer2
        supervision_layer2 = [[1, [1]], [2, [-2]], [4, [4]]]
        loss_attn_layer2 = attn_loss(attn_map2, supervision_layer2, criterion, scale=5000)
        

        # Attention Loss
        loss_attn = loss_attn_layer1 + loss_attn_layer2
        loss = 0

        if stage >= 1:
            loss2 = criterion(ee_pos_pred, ee_pos)

            supervision_layer3 = [[0, [0]], [1, [0, 2, 3]], [2, [2, 3]], [4, [4]]]
            loss_attn_layer3 = attn_loss(attn_map3, supervision_layer3, criterion, scale=5000)

            writer.add_scalar('train loss ee pos from joints', loss2.item(), global_step=epoch_idx * len(data_loader) + idx)

            loss = loss2
            loss_attn = loss_attn + loss_attn_layer3


        if stage >= 2:
            # Attention Supervision for Target Pos, EEF Pos, Command
            traj_attn = attn_map4[:, 4, 0] + attn_map4[:, 4, 1] + attn_map4[:, 4, 2] + attn_map4[:, 4, -1] + attn_map4[:, 4, -2]
            loss_traj_attn = criterion(traj_attn, torch.ones(attn_map4.shape[0], 1, dtype=torch.float32).to(device)) * 5000
            loss_attn = loss_attn + loss_traj_attn

            # Only training on xyz, ignoring rpy
            # For trajectory, use a pre-defined weight matrix to indicate the importance of the trajectory points
            trajectory_pred = trajectory_pred
            ee_traj = ee_traj
            weight_matrix = torch.tensor(np.array([1 ** i for i in range(ee_traj.shape[-1])]), dtype=torch.float32) + torch.tensor(np.array([0.9 ** i for i in range(ee_traj.shape[-1]-1, -1, -1)]), dtype=torch.float32)
            weight_matrix = weight_matrix.unsqueeze(0).unsqueeze(1).repeat(ee_traj.shape[0], ee_traj.shape[1], 1).cuda()
            loss4 = (criterion2(trajectory_pred, ee_traj) * weight_matrix).sum() / (weight_matrix).sum()
            writer.add_scalar('train loss traj', loss4.item(), global_step=epoch_idx * len(data_loader) + idx)
            loss = loss + loss4
            print('loss traj', loss4.item())

        loss = loss + loss_attn


        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # Log and print
        writer.add_scalar('train loss', loss.item(), global_step=epoch_idx * len(data_loader) + idx)
        if stage == 0:
            print(f'epoch {epoch_idx}, step {idx}, stage {stage}, l_all {loss.item():.2f}')
        elif stage == 1:
            print(f'epoch {epoch_idx}, step {idx}, stage {stage}, l_all {loss.item():.2f}, l2 {loss2.item():.2f}')
        else:
            print(f'epoch {epoch_idx}, step {idx}, stage {stage}, l_all {loss.item():.2f}, l2 {loss2.item():.2f}, l4 {loss4.item():.2f}')


        # Save checkpoint
        if save_ckpt:
            if not os.path.isdir(os.path.join(ckpt_path, name)):
                os.mkdir(os.path.join(ckpt_path, name))
            if global_step % 10000 == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(checkpoint, os.path.join(ckpt_path, name, f'{global_step}.pth'))

        # if global_step == 50:
        #     scheduler.step()

        # elif global_step == 100:
        #     scheduler.step()
    return stage


def test(writer, name, epoch_idx, data_loader, model, criterion, train_dataset_size, stage, print_attention_map=False, train_split=False):
    with torch.no_grad():
        model.eval()
        error_trajectory = 0
        error_gripper = 0
        loss5_accu = 0
        idx = 0
        error_displacement = 0
        error_ee_pos = 0
        error_joints_prediction = 0
        num_datapoints = 0
        num_trajpoints = 0
        num_grippoints = 0
        criterion2 = nn.MSELoss(reduction='none')


        for idx, (img, joint_angles, ee_pos, ee_traj, length, sentence, joint_angles_traj) in enumerate(data_loader):
            global_step = epoch_idx * len(data_loader) + idx

            # Prepare data
            img = img.to(device)
            joint_angles = joint_angles.to(device)
            ee_pos = ee_pos.to(device)
            ee_traj = ee_traj.to(device)
            length = length.to(device)
            sentence = sentence.to(device)
            joint_angles_traj = joint_angles_traj.to(device)
            ee_traj = torch.cat((ee_traj, joint_angles_traj[:, -1:, :]), axis=1)

            # Forward pass
            if stage == 0:
                attn_map, attn_map2 = model(img, joint_angles, sentence, None, stage)
            elif stage == 1:
                target_xy_pred, ee_pos_pred, attn_map, attn_map2, attn_map3 = model(img, joint_angles, sentence, None, stage)
            else:
                target_xy_pred, ee_pos_pred, attn_map, attn_map2, attn_map3, attn_map4, trajectory_pred = model(img, joint_angles, sentence, None, stage)



            if stage >= 1:
                ee_pos = ee_pos.detach().cpu()
                ee_pos_pred = ee_pos_pred.detach().cpu()
                error_ee_pos_this_time = torch.sum(((ee_pos_pred[:, :3] - ee_pos[:, :3])) ** 2, axis=1) ** 0.5
                error_ee_pos += error_ee_pos_this_time.sum()

            if stage >= 2:
                trajectory_pred = trajectory_pred
                ee_traj = ee_traj
                # Only training on xyz, ignoring rpy
                # loss1 = criterion2(trajectory_pred, ee_traj).sum() / mask.sum()

                trajectory_pred = trajectory_pred.detach().cpu().transpose(2, 1)
                ee_traj = ee_traj.detach().cpu().transpose(2, 1)
                
                error_trajectory_this_time = torch.sum(((trajectory_pred[:, :, :3] - ee_traj[:, :, :3])) ** 2, axis=2) ** 0.5
                error_trajectory_this_time = torch.sum(error_trajectory_this_time)
                error_trajectory += error_trajectory_this_time
                num_trajpoints += ee_traj.shape[0] * ee_traj.shape[1]

                # error_gripper_this_time = torch.sum(((trajectory_pred[:, :, 3:] - ee_traj[:, :, 3:]) * torch.tensor([std_joints[-1]])) ** 2, axis=2) ** 0.5
                # error_gripper_this_time = torch.sum(error_gripper_this_time)
                # error_gripper += error_gripper_this_time
                # num_grippoints += torch.sum(mask[:, 3, :]) / mask.shape[1]

            idx += 1

            # Print
            # print(f'test: epoch {epoch_idx}, step {idx}, loss5 {loss5.item():.2f}')
            if stage >= 2:
                print(idx, f'err traj {(error_trajectory / num_trajpoints).item():.4f}')

        # Log
        if not train_split:
            if stage >= 1:
                writer.add_scalar('test error_ee_pos', error_ee_pos / num_datapoints, global_step=epoch_idx * train_dataset_size)
            if stage >= 2:
                writer.add_scalar('test error_trajectory', error_trajectory / num_trajpoints, global_step=epoch_idx * train_dataset_size)
        else:
            if stage >= 1:
                writer.add_scalar('train_split error_ee_pos', error_ee_pos / num_datapoints, global_step=epoch_idx * train_dataset_size)
            if stage >= 2:
                writer.add_scalar('train_split error_trajectory', error_trajectory / num_trajpoints, global_step=epoch_idx * train_dataset_size)


def main(writer, name, batch_size=32):
    train_set_path = sys.argv[1]
    val_set_path = sys.argv[2]
    ckpt_path = sys.argv[3]
    save_ckpt = True
    supervised_attn = True
    curriculum_learning = True
    ckpt = None

    # load model
    model = Backbone(img_size=224, embedding_size=192, num_traces_in=7, num_traces_out=10, num_weight_points=24, input_nc=3)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt), strict=True)

    model = model.to(device)

    # load data
    data_dirs = [os.path.join(train_set_path, x) for x in os.listdir(train_set_path)]
    val_data_dirs = [os.path.join(val_set_path, x) for x in os.listdir(val_set_path)]
    
    dataset_train_dmp = DMPDatasetEERandTarXYLang(data_dirs, length_total=24)
    data_loader_train_dmp = torch.utils.data.DataLoader(dataset_train_dmp, batch_size=batch_size,
                                          shuffle=True, num_workers=8,
                                          collate_fn=pad_collate_xy_lang)
    dataset_test_dmp = DMPDatasetEERandTarXYLang(val_data_dirs, length_total=24)
    data_loader_test_dmp = torch.utils.data.DataLoader(dataset_test_dmp, batch_size=batch_size,
                                          shuffle=True, num_workers=8,
                                          collate_fn=pad_collate_xy_lang)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    print('loaded')

    # train n epoches
    loss_stage = 0
    for i in range(0, 800):
        whether_test = ((i % 10) == 0)
        if loss_stage <= 1:
            loss_stage = train(writer, name, i, data_loader_train_dmp, model, optimizer, scheduler,
                criterion, ckpt_path, save_ckpt, loss_stage, supervised_attn=supervised_attn, curriculum_learning=curriculum_learning, print_attention_map=False)
            if whether_test:
                test(writer, name, i + 1, data_loader_test_dmp, model, criterion, len(data_loader_train_dmp), loss_stage, print_attention_map=False)
                # test(writer, name, i + 1, data_loader_train_split, model, criterion, len(data_loader_train), loss_stage, print_attention_map=False, train_split=True)
        else:
            loss_stage = train(writer, name, i, data_loader_train_dmp, model, optimizer, scheduler,
                criterion, ckpt_path, save_ckpt, loss_stage, supervised_attn=supervised_attn, curriculum_learning=curriculum_learning, print_attention_map=False)
            if whether_test:
                test(writer, name, i + 1, data_loader_test_dmp, model, criterion, len(data_loader_train_dmp), loss_stage, print_attention_map=False)
                    # test(writer, name, i + 1, data_loader_train_split_dmp, model, criterion, len(data_loader_train_dmp), loss_stage, print_attention_map=False, train_split=True)
        if i > 2 and i <= 4:
            loss_stage = 1
        elif i > 4:
            loss_stage = 2


if __name__ == '__main__':
    name = 'modattn-pick-duck'
    writer = SummaryWriter('runs/' + name)
    main(writer, name)
