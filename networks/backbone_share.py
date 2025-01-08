import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
#from nets.transformer_noffn import Transformer
#from nets.transformer_noffn import Transformer
from torchvision import ops
from networks.plugs.SpChAttn import CBAM
import torch

#from nets.cbam import SpatialGate
from copy import deepcopy
#预训练模型权重，用来初始化网络
model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

class FPN(nn.Module):
    def __init__(self, pretrained=True):
        super(FPN, self).__init__()
        self.in_planes = 64

        print('basline')

        #初始化手部网络resnet_hand
        resnet_hand = resnet50(pretrained=pretrained)

        #resnet_obj = resnet50(pretrained=pretrained)
        # 初始化物体网络resnet_obj，与手部采用一样的网络
        resnet_obj = deepcopy(resnet_hand)# 深拷贝，与resnet_hand是相互独立的

        #1x1的卷积层，用来改变通道数，将通道数从2048降到256
        self.toplayer_h = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        #同理
        self.toplayer_o = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # ----------hand-resNet50-stage0----------------对应论文里的stage0阶段
        self.layer0_h = nn.Sequential(resnet_hand.conv1, resnet_hand.bn1, resnet_hand.leakyrelu, resnet_hand.maxpool)
        # ----------hand-resNet50-stage1----------------
        self.layer1_h = nn.Sequential(resnet_hand.layer1)
        # ----------hand-resNet50-stage2----------------2和3阶段独立提取手和物体特征，0、1和4阶段卷积层共享参数
        self.layer2_h = nn.Sequential(resnet_hand.layer2)
        # ----------hand-resNet50-stage3----------------
        self.layer3_h = nn.Sequential(resnet_hand.layer3)
        # ----------hand-resNet50-stage4----------------
        self.layer4_h = nn.Sequential(resnet_hand.layer4)

        # 对于obj，stage0、stage1、stage4阶段都是公用的，stage2、stage3是独立的！
        #self.layer0_o = nn.Sequential(resnet_obj.conv1, resnet_obj.bn1, resnet_obj.leakyrelu, resnet_obj.maxpool)
        #self.layer1_o = nn.Sequential(resnet_obj.layer1)
        # ----------obj-resNet50-stage2----------------
        self.layer2_o = nn.Sequential(resnet_obj.layer2)
        # ----------obj-resNet50-stage3----------------
        self.layer3_o = nn.Sequential(resnet_obj.layer3)
        #self.layer4_o = nn.Sequential(resnet_obj.layer4)


        # Smooth layers
        #这层卷积层的目的是在不改变特征图尺寸的前提下，通过学习更复杂的特征来进行特征图的平滑处理
        #self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        #self.smooth2_h = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_h = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        #self.smooth2_o = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        #这层卷积层的目的是在不改变特征图尺寸的前提下，通过学习更复杂的特征来进行特征图的平滑处理
        self.smooth3_o = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        # 定义用于特征融合的卷积层
        # 这些层将用于处理来自不同深度的特征图，将其转换为具有相同通道数的特征图
        # 以便进行后续的特征融合操作
        self.latlayer1_h = nn.Conv2d( 1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2_h = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3_h = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        # 定义用于特征融合的卷积层
        # 这些层与上一组层类似，用于处理不同深度的特征图
        # 目的是并行处理特征图，为最终的特征融合做准备
        self.latlayer1_o = nn.Conv2d( 1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2_o = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3_o = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        #cbam空间通道注意力模块
        #新加部分
        # self.cbam3_h=CBAM(512)
        # self.cbam3_o=CBAM(512)
        # self.cbam4_h=CBAM(1024)
        # self.cbam4_o=CBAM(1024)
        # self.cbam5_h=CBAM(2048)
        # self.cbam5_o=CBAM(2048)


    def _upsample_add(self, x, y):
        """
        对两个特征图进行上采样相加操作。

        该函数旨在对两个特征图x和y进行操作，其中y的尺寸较小。操作的步骤是首先根据y的尺寸对x进行上采样，
        然后将上采样后的x与y进行相加，以实现特征融合。这一操作常用于特征金字塔网络中的特征融合步骤，
        以结合不同层级的特征信息。

        参数:
        x (Tensor): 较大尺寸的特征图，需要对其进行上采样。
        y (Tensor): 较小尺寸的特征图，作为参考进行上采样的目标尺寸。

        返回:
        Tensor: 上采样后x与y相加的结果，实现了特征融合。
        """
        # 获取较小特征图y的尺寸信息
        _, _, H, W = y.size()
        # 将较大特征图x上采样到与y相同的尺寸，然后与y相加实现特征融合
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        c1_h = self.layer0_h(x)
        #c1_o = self.layer0_h(x)
        #print('c1_h的维度为：',c1_h.shape)

        c2_h = self.layer1_h(c1_h)
        #c2_o = c2_h
        #c2_o = self.layer1_h(c1_o)
        #print('c2_h的维度为：',c2_h.shape)
    
        c3_h = self.layer2_h(c2_h)
        c3_o = self.layer2_o(c2_h)
        #新加部分
        # c3_h = self.cbam3_h(c3_h)
        # c3_o = self.cbam3_o(c3_o)
        #print('c3_h的维度为：',c3_h.shape)
        #print('c3_o的维度为：',c3_o.shape)
        
        
  
        c4_h = self.layer3_h(c3_h)
        c4_o = self.layer3_o(c3_o)
        #新加部分
        # c4_h = self.cbam4_h(c4_h)
        # c4_o = self.cbam4_o(c4_o)
        #print('c4_h的维度为：',c4_h.shape)
        #print('c4_o的维度为：',c4_o.shape)
 
        c5_h = self.layer4_h(c4_h)
        c5_o = self.layer4_h(c4_o)
        #新加部分
        # c5_h = self.cbam5_h(c5_h)
        # c5_o = self.cbam5_o(c5_o)
        #print('c5_h的维度为：',c5_h.shape)
        #print('c5_o的维度为：',c5_o.shape)
    
        # Top-down
        p5_h = self.toplayer_h(c5_h)
        #print('p5_h的维度为：',p5_h.shape)
        p4_h = self._upsample_add(p5_h, self.latlayer1_h(c4_h))
        #print('p4_h的维度为：',p4_h.shape)     
        p3_h = self._upsample_add(p4_h, self.latlayer2_h(c3_h))
        #print('p3_h的维度为：',p3_h.shape)
        p2_h = self._upsample_add(p3_h, self.latlayer3_h(c2_h))
        #print('p2_h的维度为：',p2_h.shape)


        p5_o = self.toplayer_o(c5_o)
        #print('p5_o的维度为：',p5_o.shape)
        p4_o = self._upsample_add(p5_o, self.latlayer1_o(c4_o))
        #print('p4_o的维度为：',p4_o.shape)
        p3_o = self._upsample_add(p4_o, self.latlayer2_o(c3_o))
        #print('p3_o的维度为：',p3_o.shape)
        p2_o = self._upsample_add(p3_o, self.latlayer3_o(c2_h))
        #print('p2_o的维度为：',p2_o.shape)
        # Smooth
        #p4 = self.smooth1(p4)
        #p3_h = self.smooth2(p3_h)

        p2_h = self.smooth3_h(p2_h)
        p2_o = self.smooth3_o(p2_o)
        #print(p2.shape)
        

        
        return p2_h , p2_o

class FPN_18(nn.Module):
    def __init__(self, pretrained=True):
        super(FPN_18, self).__init__()
        self.in_planes = 64

        resnet_hand = resnet18(pretrained=pretrained)

        #resnet_obj = resnet18(pretrained=pretrained)
        resnet_obj = deepcopy(resnet_hand)

        self.toplayer_h = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        self.toplayer_o = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        self.layer0_h = nn.Sequential(resnet_hand.conv1, resnet_hand.bn1, resnet_hand.leakyrelu, resnet_hand.maxpool)
        self.layer1_h = nn.Sequential(resnet_hand.layer1)
        self.layer2_h = nn.Sequential(resnet_hand.layer2)
        self.layer3_h = nn.Sequential(resnet_hand.layer3)
        self.layer4_h = nn.Sequential(resnet_hand.layer4)

       # self.layer0_o = nn.Sequential(resnet_obj.conv1, resnet_obj.bn1, resnet_obj.leakyrelu, resnet_obj.maxpool)
        #self.layer1_o = nn.Sequential(resnet_obj.layer1)
        self.layer2_o = nn.Sequential(resnet_obj.layer2)
        self.layer3_o = nn.Sequential(resnet_obj.layer3)
       # self.layer4_o = nn.Sequential(resnet_obj.layer4)


        # Smooth layers
        #self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_h = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_h = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.smooth2_o = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_o = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1_h = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2_h = nn.Conv2d( 128, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3_h = nn.Conv2d( 64, 256, kernel_size=1, stride=1, padding=0)


        self.latlayer1_o = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2_o = nn.Conv2d( 128, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3_o = nn.Conv2d( 64, 256, kernel_size=1, stride=1, padding=0)


        self.pool_h = nn.AvgPool2d(2, stride=2)
        self.pool_o = nn.AvgPool2d(2, stride=2)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        c1_h = self.layer0_h(x)
        #c1_o = self.layer0_o(x)

        c2_h = self.layer1_h(c1_h)
       # c2_o = self.layer1_o(c1_o)
    
        c3_h = self.layer2_h(c2_h)
        c3_o = self.layer2_o(c2_h)
  
        c4_h = self.layer3_h(c3_h)
        c4_o = self.layer3_o(c3_o)
 
        c5_h = self.layer4_h(c4_h)
        c5_o = self.layer4_o(c4_o)
    
        # Top-down
        p5_h = self.toplayer_h(c5_h)
        p4_h = self._upsample_add(p5_h, self.latlayer1_h(c4_h))
        p3_h = self._upsample_add(p4_h, self.latlayer2_h(c3_h))
        p2_h = self._upsample_add(p3_h, self.latlayer3_h(c2_h))


        p5_o = self.toplayer_o(c5_o)
        p4_o = self._upsample_add(p5_o, self.latlayer1_o(c4_o))
        p3_o = self._upsample_add(p4_o, self.latlayer2_o(c3_o))
        p2_o = self._upsample_add(p3_o, self.latlayer3_o(c2_h))
        # Smooth


        p2_h = self.smooth3_h(p2_h)
        p2_o = self.smooth3_o(p2_o)
       
        
        return p2_h , p2_o


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.mean(3).mean(2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model Encoder"""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url("https://download.pytorch.org/models/resnet50-19c8e357.pth"))
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.leakyrelu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyrelu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.leakyrelu(out)

        return out

#新加部分
if __name__ == '__main__':
    def count_parameters(model):
        num_params = sum(p.numel() for p in model.parameters())
        return num_params
    net=FPN(pretrained = False)
    input_tensor = torch.randn(1, 3, 256, 256)
    with torch.no_grad():  # 关闭梯度计算，因为这里只是测试
        output_h, output_o = net(input_tensor)
    
    print("Output hand feature map shape:", output_h.shape)
    print("Output object feature map shape:", output_o.shape)
    total_params = count_parameters(net)
    print(f"Total Parameters: {total_params}")