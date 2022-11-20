'''
Default behavior is A -> B
'''
from options.eval_options import EvalOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html, util 
import numpy as np
from PIL import Image
import sys
import os

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = EvalOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.num_test = 1 # test code only supports num_test = 1000
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    if opt.eval:
        model.eval()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.forward()

        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths

        #print(img_path)
        predicted_image = Image.fromarray(util.tensor2im(visuals['rec_A']) , 'RGB')

        if(opt.save_image):
            predicted_image.save(os.path.join(opt.results_dir, 'clean_' + img_path[0].split('/')[-1]))
        else:
            predicted_image.save(sys.stdout, 'PNG')