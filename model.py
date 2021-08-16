#coding:utf-8
import torch
import torch.nn as nn
import math  
import pdb
import torch.nn.functional as F
import scipy.io as scio

def mean_shift(dataName):	
	
    if dataName == 'CAVE':         
        band_mean = (0.0939, 0.0950, 0.0869, 0.0839, 0.0850, 0.0809, 0.0769, 0.0762, 0.0788, 0.0790, 0.0834, 
                     0.0894, 0.0944, 0.0956, 0.0939, 0.1187, 0.0903, 0.0928, 0.0985, 0.1046, 0.1121, 0.1194, 
                     0.1240, 0.1256, 0.1259, 0.1272, 0.1291, 0.1300, 0.1352, 0.1428, 0.1541) #CAVE
                     
    elif dataName == 'Harvard':
        band_mean = (0.0100, 0.0137, 0.0219, 0.0285, 0.0376, 0.0424, 0.0512, 0.0651, 0.0694, 0.0723, 0.0816,
                     0.0950, 0.1338, 0.1525, 0.1217, 0.1187, 0.1337, 0.1481, 0.1601, 0.1817, 0.1752, 0.1445, 
                     0.1450, 0.1378, 0.1343, 0.1328, 0.1303, 0.1299, 0.1456, 0.1433, 0.1303) #Hararvd 

    return band_mean
	
                        
def _to_4d_tensor(x, depth_stride=None):
    """Converts a 5d tensor to 4d by stacking
    the batch and depth dimensions."""
    x = x.transpose(0, 2)  # swap batch and depth dimensions: NxCxDxHxW => DxCxNxHxW
    if depth_stride:
        x = x[::depth_stride]  # downsample feature maps along depth dimension
    depth = x.size()[0]
    x = x.permute(2, 0, 1, 3, 4)  # NxCxDxHxW => DxNxCxHxW
    x = torch.split(x, 1, dim=0)  # split along batch dimension: NxDxCxHxW => N*[1xDxCxHxW]
    x = torch.cat(x, 1)  # concatenate along depth dimension: N*[1xDxCxHxW] => 1x(N*D)xCxHxW
    x = x.squeeze(0)  # 1x(N*D)xCxHxW => (N*D)xCxHxW
    return x, depth


def _to_5d_tensor(x, depth):
    """Converts a 4d tensor back to 5d by splitting
    the batch dimension to restore the depth dimension."""
    x = torch.split(x, depth)  # (N*D)xCxHxW => N*[DxCxHxW]
    x = torch.stack(x, dim=0)  # re-instate the batch dimension: NxDxCxHxW
    x = x.transpose(1, 2)  # swap back depth and channel dimensions: NxDxCxHxW => NxCxDxHxW
    return x
 
                           
class twoBlock(nn.Module):
    def __init__(self, wn,  n_feats=64): 
        super(twoBlock, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.conv_one = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))
        self.conv_two = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1))) 

                
    def forward(self, x):
        out = self.conv_one(x)
        out = self.relu(out)
        out = self.conv_two(out)
        out = torch.add(out, x)
        return out                       
         
class threeBlock(nn.Module):
    def __init__(self, wn, n_feats = 64 , fusionWay = 'add'):
        super(threeBlock, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
            
        self.spaFE = wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1)))
        

        self.speFE_1 = wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0), dilation=1))                
        self.speFE_2 = wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(2,0,0), dilation=2)) 
        
                                    
    def forward(self, x): 

        out = torch.add(self.speFE_1(x), self.speFE_2(x))
        out = torch.max(out, self.spaFE(x))                               
        out = torch.add(out, x)
        
        return out  
         
                
def flow_warp(input, flow, size):
    out_h, out_w = size

    n, c, h, w = input.size()

    norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)

    h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
    w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
    grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

    grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
    grid = grid + flow.permute(0, 2, 3, 1) / norm

    output = F.grid_sample(input, grid, align_corners=False)

    return output


class edge(nn.Module):
    def __init__(self, wn, n_feats):
        super(edge, self).__init__()    	
        self.conv = wn(nn.Conv2d(n_feats*2, 2, kernel_size=(3,3), stride=1, padding=(1,1)))
                                
    def forward(self, out):       
  
        size = out.size()[2:]
        flow = F.interpolate(out, (int(size[0]/4.), int(size[1]/4.)), mode='bilinear', align_corners=False)
        flow = F.interpolate(flow, size, mode='bilinear', align_corners=False)  
        flow = self.conv(torch.cat([out, flow], 1))      
        flow = flow_warp(out, flow, size)
        
        return out - flow 
                            
class CRM(nn.Module):
    def __init__(self, wn, n_twoBlocks=2, n_feats=64, fusionWay='add'):
        super(CRM, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.threeBody1 = threeBlock(wn,  n_feats) 
        self.threeBody2 = threeBlock(wn,  n_feats) 
        
        self.edge_body = edge(wn, n_feats)
        
        twoBody = [
            twoBlock(wn,  n_feats) for _ in range(n_twoBlocks)
        ]
        
        self.twoBody = nn.Sequential(*twoBody)

        self.conv = wn(nn.Conv2d(n_feats*2, n_feats, kernel_size=(3,3), stride=1, padding=1)) 
        self.gamma = nn.Parameter(torch.ones(2))
                    
        self.Sig = nn.Sigmoid()
                                                                                                                               
    def forward(self, x):
        threeModule = self.threeBody1(x)
        threeModule = self.threeBody2(threeModule)
        threeModule = threeModule + x
                
        twoModule, depth = _to_4d_tensor(threeModule)  # 2D block        
        twoModule = self.twoBody[0](twoModule)             
        twoModule = torch.cat([self.gamma[0]*self.edge_body(twoModule), self.gamma[1]*twoModule], 1)
        twoModule = self.conv(twoModule)      
        
        twoModule = self.relu(twoModule)
        twoModule = self.twoBody[1](twoModule)          
        
        out = _to_5d_tensor(twoModule, depth) + threeModule                    
        
        out = torch.add(out, x)
        
        return threeModule, twoModule,  out
	    
                                                                                                                                                            
class MDFL(nn.Module):
    def __init__(self, args):
        super(MDFL, self).__init__()
        
        scale = args.upscale_factor # define hyperparameter
        self.n_colors = args.n_colors
        n_feats = args.n_feats          
        n_twoBlocks = args.n_twoBlocks                     
        n_crm = args.n_crm 
        fusionWay = 'max' #args.fusionWay
        self.dualfusion = args.dualfusion
        
        band_mean = mean_shift(args.datasetName)   # compute mean_shfit       
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.band_mean = torch.autograd.Variable(torch.FloatTensor(band_mean)).view([1, self.n_colors, 1, 1])
        self.relu = nn.ReLU(inplace=True)       
        
        ## define head
        head = []
        head.append(wn(nn.Conv3d(1, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1))))
        head.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))))

        self.head = nn.Sequential(*head)            
        self.nearest = nn.Upsample(scale_factor=scale, mode='nearest')        
        
        ## define body
        self.crm1 = CRM(wn, n_twoBlocks, n_feats, fusionWay)      
        self.crm2 = CRM(wn, n_twoBlocks, n_feats, fusionWay)
        self.crm3 = CRM(wn, n_twoBlocks, n_feats, fusionWay)        
        self.crm4 = CRM(wn, n_twoBlocks, n_feats, fusionWay) 
        
        self.gamma_two = nn.Parameter(torch.ones(n_crm))  
        self.gamma_three = nn.Parameter(torch.ones(n_crm))                   
        self.compress_two = wn(nn.Conv2d(n_feats*n_crm, n_feats, kernel_size=(1,1), stride=1)) 
        self.compress_three = wn(nn.Conv3d(n_feats*n_crm, n_feats, kernel_size=(1,1,1), stride=1))                 

        if self.dualfusion == 'concat':  
            self.gamma_dual = nn.Parameter(torch.ones(2))            
            self.reduceD = wn(nn.Conv3d(n_feats*2, n_feats, kernel_size=(1,1,1), stride=1))           
                            
        ## define tail                         
        tail = []
        tail.append(wn(nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(3,2+scale,2+scale), stride=(1,scale,scale), padding=(1,1,1)))) 
        tail.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1))))
        tail.append(wn(nn.Conv3d(n_feats, 1, kernel_size=(3,1,1), stride=1, padding=(1,0,0))))
        self.tail = nn.Sequential(*tail)

                                                            
    def forward(self, x):
        x = x - self.band_mean.cuda() 
        z = self.nearest(x)                          
        x = x.unsqueeze(1)      
        ## initial feature extraction
        x = self.head(x)
        s = x

        ## deep fature extraction        
        threeModule = []
        twoModule = []        
        m, n,  x = self.crm1(x)
        threeModule.append(self.gamma_three[0]*m)
        twoModule.append(self.gamma_two[0]*n)   
             
        m, n, x = self.crm2(x)
        threeModule.append(self.gamma_three[1]*m)
        twoModule.append(self.gamma_two[1]*n) 
        
        m, n,  x = self.crm3(x)
        threeModule.append(self.gamma_three[2]*m)
        twoModule.append(self.gamma_two[2]*n) 
        
        m, n, _ = self.crm4(x)
        threeModule.append(self.gamma_three[3]*m)
        twoModule.append(self.gamma_two[3]*n) 

        del m, n,  x 

        threeModule = torch.cat(threeModule, 1)  
        threeModule = self.compress_three(threeModule) 
        
        twoModule = torch.cat(twoModule, 1)
        twoModule = self.compress_two(twoModule)
                                 
        
       # dual-channel fusion
        if self.dualfusion == 'concat':
            threeModule = torch.cat([self.gamma_dual[0]*_to_5d_tensor(twoModule, self.n_colors), self.gamma_dual[1]*threeModule], 1)   
            threeModule = self.reduceD(threeModule)               
                
        del twoModule
                               
        threeModule = torch.add(threeModule, s)
        
        ## image restruction    
        threeModule = self.tail(threeModule)               
        threeModule = threeModule.squeeze(1)  
        threeModule = torch.add(threeModule, z)           
        threeModule = threeModule + self.band_mean.cuda()   
        return threeModule  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
