import torch 
import torch.nn as nn
import torch.nn.functional as F

EDSR_PATH = r"../models/EDSR_SRx2.pth"

class Supernet(nn.Module):
    def __init__(self,filters):
        super(Supernet, self).__init__()

        f = filters 
        self.conv1 = nn.Conv2d(in_channels=220, 
                            out_channels=f[0],
                            kernel_size=3,
                            stride=1, padding=1,
                            padding_mode='reflect'
                              )
        
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(f[0])

        self.conv2 = nn.Conv2d(in_channels=f[0]
                               ,out_channels=f[1], 
                               kernel_size=3, 
                               stride=1, 
                               padding=1, 
                               padding_mode='reflect'
                               )
        
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(f[1])

        self.conv3 = nn.Conv2d(in_channels=f[1],
                                out_channels=f[2],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                padding_mode='reflect')

        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(f[2])
        
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv4 = nn.Conv2d(in_channels=f[2],
                                out_channels=f[3],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                padding_mode='reflect')
        
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(f[3])

        self.conv5 = nn.Conv2d(in_channels=f[3],
                                out_channels=f[4],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                padding_mode='reflect')
        
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(f[4])

    def forward(self, input):

        output = self.bn1(self.relu1(self.conv1(input)))

        output = self.bn2(self.relu2(self.conv2(output)))

        output = self.bn3(self.relu3(self.conv3(output)))
        
        output = self.upscale(output)
        
        output = self.bn4(self.relu4(self.conv4(output)))

        output = self.bn5(self.relu5(self.conv5(output)))
        
        return output


def load_edsr_weights(model):

    edsr = torch.load(EDSR_PATH)
    edsr_state_dict = {}
    for key, value in edsr.state_dict().items():
        if "mean" in key:
            continue
        else:
            edsr_state_dict[key] = value
    
    new_state_dict = model.state_dict()
    for key, value in zip(new_state_dict.keys(), edsr_state_dict.values()):
        if "conv0" in key or "conv3" in key:
            continue
        new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    return model

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * 1.0
        out += x
        
        return out

class Body(nn.Module):
    def __init__(self, num_blocks, channels):
        super(Body, self).__init__()
        
        self.blocks = nn.ModuleList([ResidualBlock(channels, channels) for _ in range(num_blocks)])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        return x

class AdaptEDSR(nn.Module):
   
    def __init__(self, num_blocks, ResBlocks_channels):
        super(AdaptEDSR, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=220,
                            out_channels=ResBlocks_channels,
                            kernel_size=3,
                            stride=1, padding=1)

        self.body = Body(num_blocks, ResBlocks_channels)

        self.conv1 = nn.Conv2d(in_channels=ResBlocks_channels, 
                            out_channels=ResBlocks_channels,
                            kernel_size=3,
                            stride=1, padding=1)

        self.conv2 = nn.Conv2d(in_channels=ResBlocks_channels, 
                            out_channels=ResBlocks_channels * 4,
                            kernel_size=3,
                            stride=1, padding=1)
        
        self.upscale = nn.PixelShuffle(upscale_factor=2)

        self.conv3 = nn.Conv2d(in_channels=ResBlocks_channels, 
                            out_channels=220,
                            kernel_size=3,
                            stride=1, padding=1)

    def forward(self, input):

        x = self.conv0(input)
        y = self.body(x) + x
        # x = self.conv1(y) 
        x = self.conv2(y) #edité le 10/05/2024 (turn it to comment)
        x = self.upscale(x)
        x = self.conv3(x)

        return x
    

if __name__ == "__main__":

    # model = Supernet(filters=[700, 500, 400, 300, 220])
    model = AdaptEDSR(num_blocks=16, ResBlocks_channels=64)
    y = torch.rand((1,220,8,8))
    total_params = sum(
	param.numel() for param in model.parameters())
    print(total_params)
    z= model(y)
    print(z.shape)

    
    edsr = torch.load(EDSR_PATH)
    print(edsr)
    print(len(edsr.state_dict().keys()))
    print(len(model.state_dict().keys()))

    edsr_state_dict = {}
    for key, value in edsr.state_dict().items():
        if "mean" in key:
            continue
        else:
            edsr_state_dict[key] = value
    
    new_state_dict = model.state_dict()
    for key, value in zip(new_state_dict.keys(), edsr_state_dict.values()):
        if "conv0" in key or "conv3" in key:
            continue
        new_state_dict[key] = value

    print(torch.equal(model.state_dict()["body.blocks.6.conv1.weight"], edsr.state_dict()["model.body.6.body.0.weight"]))

    model.load_state_dict(new_state_dict)

    print(torch.equal(model.state_dict()["body.blocks.6.conv1.weight"], edsr.state_dict()["model.body.6.body.0.weight"]))