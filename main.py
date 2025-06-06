# MAMRN train 
import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from model import MAMRNet

import RS, Folder
import utils
import skimage.color as sc
import random
from utils import *
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Training settings
parser = argparse.ArgumentParser(description="network")
parser.add_argument("--batch_size", type=int, default=128,
                    help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=16,
                    help="testing batch size")
parser.add_argument("-nEpochs", type=int, default=500,
                    help="number of epochs to train")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="Learning Rate. Default=0.0001")
parser.add_argument("--step_size", type=int, default=50,
                    help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.5,
                    help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True,
                    help="use cuda")
parser.add_argument("--resume", default="", type=str,
                    help="path to checkpoint")
parser.add_argument("--start-epoch", default=1, type=int,
                    help="manual epoch number")
parser.add_argument("--threads", type=int, default=0,
                    help="number of threads for data loading")
parser.add_argument("--root", type=str, default=r"  ",   #your path
                    help='dataset directory')
parser.add_argument("--n_train", type=int, default=2000,
                    help="number of training set")
parser.add_argument("--n_val", type=int, default=16,
                    help="number of validation set")
parser.add_argument("--test_every", type=int, default=2000)
parser.add_argument("--scale", type=int, default=6,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=96,
                    help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1,
                    help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=1,
                    help="number of color channels to use")
parser.add_argument("--pretrained", default="", type=str,
                    help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--isY", action="store_true", default=False)
parser.add_argument("--ext", type=str, default='.npy')
parser.add_argument("--phase", type=str, default='train')


args = parser.parse_args()
print(args)
torch.backends.cudnn.benchmark = True
# random seed
seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)
print("Ramdom Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')

print("===> Loading datasets")

trainset = RS.rs(args)
testset = Folder.DatasetFromFolderVal(r"  ",     #your MODIS path
                                       r"  ",args.scale)    #your CLDIS path
training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
testing_data_loader = DataLoader(dataset=testset, num_workers=args.threads, batch_size=args.testBatchSize,
                                 shuffle=False)

print("===> Building models")
args.is_train = True

model = MAMRNet(res_down=True, n_resblocks=1, bilinear=0)

l1_criterion =nn.L1Loss()
l2_criterion=nn.MSELoss()

print("===> Setting GPU")
if cuda:
    model = model.to(device)
    l1_criterion = l1_criterion.to(device)
    l2_criterion = l2_criterion.to(device)

if args.pretrained:
    if os.path.isfile(args.pretrained):
        print("===> loading models '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        new_state_dcit = OrderedDict()
        for k, v in checkpoint.items():
            if 'module' in k:
                name = k[7:]
            else:
                name = k
            new_state_dcit[name] = v
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dcit.items() if k in model_dict}

        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        model.load_state_dict(pretrained_dict, strict=True)

    else:
        print("===> no models found at '{}'".format(args.pretrained))


print("===> Setting Optimizer")

optimizer = optim.Adam(model.parameters(), lr=args.lr)

writer=SummaryWriter()

def train(epoch):
    model.train()
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):

        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]

        optimizer.zero_grad()
        sr_tensor = model(lr_tensor)

        loss_l1 = l1_criterion(sr_tensor, hr_tensor)
        loss_l2 = l2_criterion(sr_tensor, hr_tensor)
        loss_sr = 10*loss_l1 + loss_l2
        loss=loss_sr

        loss.backward()
        optimizer.step()
        if iteration % 200 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                                  loss_sr.item()))

            writer.add_scalar('Loss',loss_sr.item())
    writer.close()


def valid():
    model.eval()

    rmse, psnr_value, ssim_value = 0, 0, 0
    for batch in testing_data_loader:
        lr_tensor, hr_tensor = batch[0], batch[1]

        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

        with torch.no_grad():
            pre = model(lr_tensor)

        sr_img = utils.tensor2np(pre.detach()[0])
        gt_img = utils.tensor2np(hr_tensor.detach()[0])

        crop_size = args.scale
        cropped_sr_img = utils.shave(sr_img, crop_size)
        cropped_gt_img = utils.shave(gt_img, crop_size)

        if args.isY is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:,:, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img

        rmse += (utils.calculate_metrics(im_pre, im_label))[0]
        psnr_value += (utils.calculate_metrics(im_pre, im_label))[1]
        ssim_value += (utils.calculate_metrics(im_pre, im_label))[2]

    print("===> Valid:  RMSE: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}".format(rmse / len(testing_data_loader),
                                                                        psnr_value / len(testing_data_loader),
                                                                        ssim_value / len(testing_data_loader)))

def save_checkpoint(epoch):
    model_folder = "checkpoint_x{}/".format(args.scale)
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


print("===> Training")
print_network(model)
for epoch in range(args.start_epoch, args.nEpochs + 1):
    valid()
    train(epoch)
    save_checkpoint(epoch)
