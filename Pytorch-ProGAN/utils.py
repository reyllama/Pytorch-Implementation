import os
import shutil # High-level file operation : copy, delete, etc.
import torch
from apex.amp._amp_state import _amp_state

def save_checkpoint(state, save_path, is_best=False, max_keep=None):
    torch.save(state, save_path)

    # max_keep
    save_dir = os.path.dirname(save_path) # returns directory, excluding the file name
    list_path = os.path.join(save_dir, 'latest_checkpoint')
    save_path = os.path.basename(save_path) # returns the basename <-> abspath. (ex) basename("C:/Python/tmp") --> tmp
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + '\n'] + ckpt_list
    else:
        ckpt_list = [save_path + "\n"]

    if max_keep is not None: # Have max_keep limit
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, "w") as f:
        f.writelines(ckpt_list)

    # Copy Best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, "best_model.ckpt"))

def load_checkpoint(ckpt_dir_or_file, load_best=False, map_location=None):
    if os.path.isdir(ckpt_dir_or_file): # If directory
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_of_file, "best_model.ckpt")
        else:
            with open(os.path.join(ckpt_dir_or_file, "latest_checkpoint")) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else: # If file
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(" [*] Loading checkpoint from %s success!" % ckpt_path)
    return ckpt

def compute_transition_value(global_step, is_transitioning, transition_iters, latest_switch):
    transition_variable = 1
    if is_transitioning:
        diff = global_step - latest_switch
        transition_variable = diff / transition_iters
    assert 0 <= transition_variable <= 1
    return transition_variable

def get_transition_value(x_old, x_new, transition_variable):
    assert x_old.shape == x_new.shape, "Old shape: {} / New shape: {}".format(x_old.shape, x_new.shape)
    return torch.lerp(x_old, x_new, transition_variable) # Linear Interpolation. Weight=transition_variable
