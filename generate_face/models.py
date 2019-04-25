import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
##############################
#           U-NET
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        model = [   nn.Conv2d(in_size, out_size, 3, stride=2, padding=1),
                    nn.ReLU( inplace=True) ]

        if normalize:
            model += [nn.BatchNorm2d(out_size, 0.8)]

        if dropout:
            model += [nn.Dropout(dropout)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        model = [   nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_size, out_size, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_size, 0.8) ]
        if dropout:
            model += [nn.Dropout(dropout)]

        self.model = nn.Sequential(*model)

    def forward(self, x, skip_input):
        x = self.model(x)
        out = torch.cat((x, skip_input), 1)
        #out = torch.add(x, skip_input)
        return out

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256, dropout=0.5)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 256, dropout=0.5)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 64)

        def conv2d(in_features, out_features, ksize=3, stride=1, padding=1):
            lst = [  nn.Conv2d(in_features, out_features, ksize, stride, padding),
                     nn.ReLU(inplace=True),
                     nn.BatchNorm2d(out_features)
                  ]
            return lst


                                  
        self.Cg = nn.Sequential(*conv2d(512, 256, ksize=3, stride=1, padding=1))

        C  = conv2d(512, 256, ksize=3, stride=1, padding=1)
        C += conv2d(256, 64, ksize=3, stride=1, padding=1)
        C += conv2d(64 , 1, ksize=3, stride=1, padding=1)
        self.C =  nn.Sequential(*C)

        cifar = []
        cifar += conv2d(in_channels, 128, ksize=3, stride=2, padding=1)
        cifar += conv2d(128, 256, ksize=3, stride=2, padding=1)
        cifar += [nn.MaxPool2d((8, 8))]
        self.cifar = nn.Sequential(*cifar)

        self.Clg_fc = nn.Sequential(
            nn.Linear(266, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True)
        )


        middle = conv2d(512, 512, ksize=3, stride=1, padding=1)
        self.middle = nn.Sequential(*middle)

        final = [   nn.Upsample(scale_factor=2),
                    nn.Conv2d(128, out_channels, 3, 1, 1),
                    nn.Tanh() ]
        self.final = nn.Sequential(*final)
      
    def calculate_vector(self,encoder,vec):
        """
         @encoder: [N, C0, W, H]
         @vec    : [N, Cl]
        """
        # MAXPolling
        GlobalMaxPool = nn.MaxPool2d(encoder.size()[2:])
        Cg_temp = self.Cg(encoder)
        Cg = GlobalMaxPool(Cg_temp)   # Cg: [N, Cg, 1, 1]
       
        # print("Cg size is : ", Cg.size())
 
        Cl = torch.unsqueeze(vec, 2)
        Cl = torch.unsqueeze(Cl, 3)
        
        size = Cl.size()
        Cgl = torch.cat([Cg, Cl], 1)
        
        # Czeros = torch.full((size[0], 512 - 266, 1, 1), 0).cuda()
        # Cgl = torch.cat([Cg, Cl, Czeros], 1)
        # Cgl = Cl
        # Cgl = Cgl.repeat(1, 1, encoder.size()[2], encoder.size()[3])

        return Cgl

    def top_k(self, x, k):
        """
            @x:      input tensor, size = [N, 1, W, H]
            @k:      int, mean the top number
            @return: lst, 
        """
        """
        MINIMUM = -1
        lst = []
        clone_tensor = x.clone()
        for _ in range(k):
            temp = []
            for i in range(x.size()[0]):
                index = torch.max(clone_tensor[i:i+1,:,:,:], dim=0, keepdim=True)
                clone_tensor[index] = MINIMUM
                temp.append(index)
            lst.append(torch.cat(temp, 0))
        """
        lst = []
        temp = torch.full((x.size()[0], 1, 8, 8), 0).cuda()
        for i in range(x.size()[0]):
            temp[i, 0, 6, 7] = 1
        lst.append(temp)
  
        temp = torch.full((x.size()[0], 1, 8, 8), 0).cuda()
        for i in range(x.size()[0]):
            temp[i, 0, 2, 7] = 1
        lst.append(temp)
        
        return lst

    def k_feature(self, x, vec, k_list):
        lst = []
        for index in k_list:
            index1 =     index.repeat(1, vec.size()[1], 1, 1)	# this is like a mask
            index2 = 1 - index.repeat(1, x.size()[1], 1, 1)     # this is like a 1 - mask
            lst.append(x * index2 + vec * index1)
        return lst

    def forward(self, x, cifar_img, vec):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        
        """ the idea that the uses the encodered cifar code to fill in the area""" 
        """
        Cg = self.cifar(cifar_img)
        Cl = torch.unsqueeze(vec, 2)
        Cl = torch.unsqueeze(Cl, 3)
        Czeros = torch.full((Cg.size()[0], 512 - 266, 1, 1), 0).cuda()
        Cgl = torch.cat([Cg, Cl, Czeros], 1)
        Cgl = Cgl.repeat(1, 1, 8, 8) 
        """
        Cgl = self.calculate_vector(d5, vec)	# get the vector we will embed in
       						# size = [N, 512, 8, 8]
        Cgl = Cgl.view(Cgl.size()[0], -1)
        Cgl = self.Clg_fc(Cgl)
        Cgl = Cgl.view(Cgl.size()[0], Cgl.size()[1], 1, 1)
        Cgl = Cgl.repeat(1, 1, 8, 8)
        # print("Cgl size ", Cgl.size())
        

        # Cglo = torch.cat([Cgl, d5], 1)


        # C = self.C(Cglo)
        k_list = self.top_k(Cgl, 2)		# get the location we will set the patch
        k_feat = self.k_feature(d5, Cgl, k_list) # set the patch into the picture

        lst = []
        for item in k_feat:
            middle = self.middle(item)
            u1 = self.up1(middle, d4)
            u2 = self.up2(u1, d3)
            u3 = self.up3(u2, d2)
            u4 = self.up4(u3, d1)
            lst.append(self.final(u4))

        return lst, k_list



##############################
#           RESNET
##############################

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.BatchNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.BatchNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, resblocks=9):
        super(GeneratorResNet, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(in_channels, 64, 7),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features
        
        def conv2d(in_features, out_features, ksize=3, stride=1, padding=1):        
            lst = [     nn.Conv2d(in_features, out_features, ksize, stride, padding),
                        nn.BatchNorm2d(out_features),
                        nn.ReLU(inplace=True)   ]
            return lst

        model += conv2d(64, 128, 3, 2, 1)
        model += conv2d(128,256, 3, 2, 1)
        model += conv2d(256,512, 3, 2, 1)
        model += conv2d(512,512, 3, 2, 1)
        model += conv2d(512,512, 3, 2, 1)
        # Here we get the output of encoder


        # Here we get the model of Residual blocks
        # Residual blocks
        for _ in range(resblocks // 2):
            model += [ResidualBlock(512)]
        self.encoder = nn.Sequential(*model)

        # Here we define the model of decoder
        
        model = conv2d(512*2+10, 512, 3, 1, 1)
        for _ in range(resblocks // 2 - 1):
            model += [ResidualBlock(512)]

        # Upsampling
        def Transposeconv2d(in_features, out_features, ksize=3, stride=1, padding=1, output_padding=1):        
            lst = [     nn.ConvTranspose2d(in_features, out_features, ksize, stride, padding, output_padding),
                        nn.BatchNorm2d(out_features),
                        nn.ReLU(inplace=True)   ]
            return lst
        model += Transposeconv2d(512,512, 3, 2, 1, 1)
        model += Transposeconv2d(512,512, 3, 2, 1, 1)
        model += Transposeconv2d(512,256, 3, 2, 1, 1)
        model += Transposeconv2d(256,128, 3, 2, 1, 1)
        model += Transposeconv2d(128, 64, 3, 2, 1, 1)

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, out_channels, 7),
                    nn.Sigmoid() ]

        self.decoder = nn.Sequential(*model)

    def calculate_vector(self,encoder,vec):
        """
         @encoder: [N, C0, W, H]
         @vec    : [N, Cl]
        """
        # MAXPolling
        GlobalMaxPool = nn.MaxPool2d(encoder.size()[2:])
        Cg = GlobalMaxPool(encoder)   # Cg: [N, Cg, 1, 1]
        
        Cl = torch.unsqueeze(vec, 2)
        Cl = torch.unsqueeze(Cl, 3)
        
        Cgl = torch.cat([Cg, Cl], 1)
        Cgl = Cgl.repeat(1, 1, encoder.size()[2], encoder.size()[3])

        return Cgl

    def C(self, x):
        """
          @x:      input tensor, size = [N, C0+Cg+Cl, W, H]
          @return: return tensor, size = [N, 1, W, H]
        """
        
        model = [
                   nn.Conv2d(x.size()[1], 32, 3, stride=1, padding=1),
                   nn.BatchNorm2d(32),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(32, 1, 3, stride=1, padding=1),
                   nn.BatchNorm2d(1),
                   nn.ReLU(inplace=True)
                ]
        model = nn.Sequential(*model).cuda()
        return model(x)

    def top_k(self, x, k):
        """
            @x:      input tensor, size = [N, 1, W, H]
            @k:      int, mean the top number
            @return: lst, 
        """
        """
        MINIMUM = -1
        lst = []
        clone_tensor = x.clone()
        for _ in range(k):
            temp = []
            for i in range(x.size()[0]):
                index = torch.max(clone_tensor[i:i+1,:,:,:], dim=0, keepdim=True)
                clone_tensor[index] = MINIMUM
                temp.append(index)
            lst.append(torch.cat(temp, 0))
        """
        lst = []
        temp = torch.full(x.size(), 0).cuda()
        for i in range(x.size()[0]):
            temp[i, 0, 6, 7] = 1
        lst.append(temp)
  
        temp = torch.full(x.size(), 0).cuda()
        for i in range(x.size()[0]):
            temp[i, 0, 2, 7] = 1
        lst.append(temp)
        
        return lst

    def k_feature(self, x, vec, k_list):
        lst = []
        for index in k_list:
            index = index.repeat(1, vec.size()[1], 1, 1)
            lst.append(torch.cat([x, vec * index], 1))
        return lst

    def forward(self, x, vec):
        encoder = self.encoder(x)

        # print("encoder size is ", encoder.size())

        Cgl = self.calculate_vector(encoder, vec)
        Cglo = torch.cat([Cgl, encoder], 1)

        C = self.C(Cglo)          # size: N * 1 * w * h
        k_list = self.top_k(C, 2)
        k_feat = self.k_feature(encoder, Cgl, k_list)

        lst = []
        for item in k_feat:
            lst.append(self.decoder(item))
        # return [self.decoder(encoder)], [k_list[0]]
        return lst, k_list 

##############################
#        Discriminator
##############################
'''
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 64, 2, False),
            *discriminator_block(64, 128, 2, True),
            *discriminator_block(128, 256, 2, True),
            *discriminator_block(256, 512, 2, True),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
'''

class CDiscriminator(nn.Module):
    def __init__(self,class_axis = 10, patch_size = 32, ch = 3):
        super(CDiscriminator,self).__init__()
        self.model = nn.Sequential(
        nn.Linear(class_axis + patch_size * patch_size * ch, 512),
        nn.ReLU(inplace = True),
        nn.Linear(512, 512),
        nn.Dropout(0.4),
        nn.ReLU(inplace = True),
        nn.Linear(512,512),
        nn.Dropout(0.4),
        nn.ReLU(inplace = True),
        nn.Linear(512,1),
        nn.Sigmoid()
        )
    def forward(self, img, label):
     
        d_in = torch.cat((img.view(img.size(0),-1),label), -1)
        out  = self.model(d_in)

        return out 

class CDCDiscriminator(nn.Module):
    def __init__(self, class_axis = 10, patch_size = 32, ch=3):
        super(CDCDiscriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(ch, 64, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(10, 64, 4, 2, 1)

        self.conv2    = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(256)

        self.conv3    = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(512)
     
        self.conv4    = nn.Conv2d(512, 1, 4, 1, 0)

    def forward(self, input, label):
        label = label.view(label.size(0), label.size(1), 1, 1)
        label = label.repeat(1, 1, input.size(2), input.size(3))

        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)

        x = torch.cat([x, y], 1)
        
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
  
        x = F.sigmoid(self.conv4(x))

        return x.view(x.size(0), 1)
