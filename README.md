
# CycleGAN

## Installation
- Clone this repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```
- For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.
- For Docker users, we provide the pre-built Docker image and Dockerfile. Please refer to our [Docker](docs/docker.md) page.

### CycleGAN train/test

CycleGANs requires square images for training, so to standardize the images, There's a little script (under scripts foleder) for resizing the images to a 512x512 square with the signature at the center and two black (transparent) strips at the top and bottom of the image.  
  
We also have to store the data folders in a particular structure. There should be sub-folders testA, testB, trainA, and trainB inside `datasets\dataset_name`. In trainA/testA, place the clean images (domainA) and in trainB/testB, place the noisy images.  
`datasets`  
&nbsp;&nbsp;&nbsp;&nbsp; |-> `dataset_name`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-> `trainA`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-> `trainB`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-> `testA`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-> `testB`   

Signature dataset must to be added to the `datasets` folder of the cloned repository.  
  
The following line could be used to train the model.  
`python train.py --dataroot ./datasets/dataset_name --name model_name --model cycle_gan`  
  
Change the `--dataroot` and `--name` to the custom dataset path and model name.  
Use `--gpu_ids` 0,1,.. to train on multiple GPUs.  
The model_name we use for testing should be consistent with the model_name we used for training as well.  
> `!python train.py --dataroot ./datasets/gan_signdata_kaggle --name gan_signdata_kaggle --model cycle_gan`  
  
**Training arguments**
By default, the model trains for 100 epochs, to train for more epochs, use `--n_epochs #epoch_count`. The model will be trained for (100 + epoch_count) epochs.  
To continue training after you stop the training, use `--continue_train` and set `--epoch_count #epoch_number`. This will resume the training from *epoch_number* epoch.
eg: `--continue_train --epoch_count 110` will resume the training from epoch 110.
  
  
The model can translate the image in both directions (noisy to clean and clean to noisy). For our use case, we have to translate from noisy to clean. So after training, copy the latest Generator(B) `/checkpoints/model_name/latest_net_G_B.pth` as `latest_net_G.pth` under `/checkpoints/model_name/latest_net_G.pth`.
Use `cp ./checkpoints/horse2zebra/latest_net_G_B.pth ./checkpoints/horse2zebra/latest_net_G.pth` (see under testing section)
  
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with train and test script

```
To see more intermediate results, check out `./checkpoints/maps_cyclegan/web/index.html`.
- Test the model:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
- The test results will be saved to a html file here: `./results/maps_cyclegan/latest_test/index.html`.

## [Training/Test Tips](docs/tips.md)
Best practice for training and testing your models.

## [Frequently Asked Questions](docs/qa.md)
Before you post a new question, please first look at the above Q & A and existing GitHub issues.

## Custom Model and Dataset
If you plan to implement custom models and dataset for your new applications, we provide a dataset [template](data/template_dataset.py) and a model [template](models/template_model.py) as a starting point.

## [Code structure](docs/overview.md)
To help users better understand and use our code, we briefly overview the functionality and implementation of each package and each module.
