import time
import os
import numpy as np
import torch
from apex import amp
import utils
from utils import load_checkpoint, save_checkpoint
from data_tools.dataloaders_v2 import load_dataset
import config_parser
from generator import Generator
from discriminator import Discriminator
from running_average_generator import AverageGenerator
import loss

"""
=============================
Reproductibility in Pytorch
=============================

random_seed를 사용하면 항상 같은 순서로 난수가 발생
Pytorch Framework에서의 Randomness는 4 part에서 유래

1. Pytorch Randomness
torch.manual_seed(random_seed)
하지만 Pytorch 함수 중 nondeterministic 함수들 존재 - atomic operation (Tensor.index_add_(), Tensor.scsatter_add_(), bincount() 등)

2. cudnn
torch.backends.cudnn.deterministic = True <--- 학습 속도 하락 가능성. 초반보다는 후반부, 배포 단계에서 활용
torch.backends.cudnn.benchmark = False

3. Numpy
np.random.seed(random_seed)

4. Python
random.seed(random_seed)
Torchvision transforms 함수는 Python random library에 의해 randomness 결정

+ torch.cuda
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # When Multi GPU
"""

if False:
    torch.manual_seed(0)
    np.random_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptinos(precision=10)
else:
    torch.backends.cudnn.benchmark = True
    # automatically find the best algorithm to use for this hardware
    # Good when input size doesn't change much, since torch finds the best algo for current config
    # When input size varies with iterations, it might take longer run time

def init_model(latent_size, start_channel_dim, image_channels):
    discriminator = Discriminator(image_channels, start_channel_dim)
    generator = Generator(start_channel_dim, image_channels, latent_size)
    return discriminator, generator

class Trainer:

    def __init__(self, config):
        self.default_device = "cpu"
        if torch.cuda.is_available():
            self.default_device = "cuda"

        # Hyperparams
        self.batch_size_schedule = config.train_config.batch_size_schedule # Reduce Batchsize as resolution rises to control memory
        self.dataset = config.dataset
        self.learning_rate = config.train_config.learning_rate
        self.running_average_generator_decay = config.models.generator.running_average_decay
        self.full_validation = config.use_full_validation
        self.load_fraction_of_dataset = config.load_fraction_of_dataset

        # Image Settings
        self.current_imsize = 4
        self.image_channels = config.models.image_channels # 3 for RGB
        self.latent_size = config.models.latent_size
        self.max_imsize = config.max_imsize # 1024 in literature

        # Logging Variables
        self.checkpoint_dir = config.checkpoint_dir
        self.model_name = self.checkpoint_dir.split("/")[-2]
        self.config_path = config.config_path
        self.global_step = 0 # At first

        # Transition Settings
        self.transition_variable = 1.
        self.transition_iters = config.train_config.transition_iters
        self.is_transitioning = False
        self.transition_step = 0
        self.start_channel_size = config.models.start_channel_size
        self.latest_switch = 0
        self.opt_level = config.train_config.amp_opt_level
        self.start_time = time.time()
        self.discriminator, self.generator = init_model(self.latent_size, self.start_channel_size, self.image_channels)
        self.init_running_average_generator()
        self.criterion = loiss.WGANLoss(self.discriminator, self.generator, self.opt_level)
        self.logger = logger.Logger(config.summaries_dir, config.generated_data_dir)
        self.num_skipped_steps = 0
        if not self.load_checkpoint(): # First time
            self.init_optimizers()

        self.batch_size = self.batch_size_schedule[self.current_imsize]
        self.logger.log_variable("stats/batch_size", self.batch_size)

        self.num_ims_per_log = config.logging.num_ims_per_log
        self.next_log_point = self.global_step
        self.num_ims_per_save_image = config.logging.num_ims_per_save_image
        self.next_image_save_point = self.global_step
        self.num_ims_per_checkpoint = config.logging.num_ims_per_checkpoint
        self.next_validation_checkpoint = self.global_step

        self.static_latent_variable = self.generator.generate_latent_variable(64)
        self.dataloader_train = load_dataset(
            self.dataset, self.batch_size, self.current_imsize, self.full_validation, self.load_fraction_of_dataset
        )

    
