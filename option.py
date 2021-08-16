import argparse
# Training settings
parser = argparse.ArgumentParser(description="Hyperspectral Image Super-Resolution")
parser.add_argument("--upscale_factor", type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--seed', type=int, default=1,  help='random seed (default: 1)')
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="maximum number of epochs to train")
parser.add_argument("--show", action="store_true", help="show Tensorboard")


parser.add_argument("--lr", type=int, default=1e-4, help="initial lerning rate")
parser.add_argument("--cuda", action="store_true", help="Use cuda")
parser.add_argument("--gpus", type=str, default='0,1,2,3', help="gpu ids (default: 0,1,2,3)")
parser.add_argument("--threads", type=int, default=12, help="number of threads for dataloader to use")
parser.add_argument("--resume", type=str, default='', help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", type=int, default=1, help="Manual epoch number (useful on restarts)")                   


parser.add_argument("--datasetName", type=str, default='CAVE', help="data name")

# Network settings
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--n_colors', type=int, default=31, help='number of bands')
parser.add_argument('--n_twoBlocks', type=int, default=2, help='number of 2D blocks')
parser.add_argument('--fusionWay', type=str, default='max', help='fusion way in 3D block')
parser.add_argument('--n_crm', type=int, default=4, help='number of CRMs')
parser.add_argument('--dualfusion', type=str, default='concat', help='number of CRMs')

# Test image
parser.add_argument('--model_name', type=str, default='checkpoint/model.pth', help='super resolution model name for test')
parser.add_argument('--method', default='MDFLSR', type=str, help='super resolution method name')
opt = parser.parse_args()
                  