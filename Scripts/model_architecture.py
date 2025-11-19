import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.nn import init
import numpy as np

class Generator(nn.Module):
    """Generador global con local enhancer para Pix2PixHD"""
    
    def __init__(self, input_nc=3, output_nc=1, ngf=64, n_downsample_global=4, 
                 n_blocks_global=9, n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.InstanceNorm2d):
        super(Generator, self).__init__()
        
        self.n_local_enhancers = n_local_enhancers
        
        ###### Red global ######
        ngf_global = ngf
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, 
                                     n_blocks_global, norm_layer)
        
        ###### Redes locales ######  
        model_local = []
        for n in range(n_local_enhancers):
            ngf_local = ngf // (2**n)
            model_local.append(LocalEnhancer(input_nc, output_nc, ngf_local, n_downsample_global, 
                                           n_blocks_global, n_blocks_local, norm_layer, n))
        
        self.model = nn.ModuleList([model_global] + model_local)
        
    def forward(self, input):
        """Forward pass del generador"""
        # Empezar con el generador global
        output_prev = self.model[0](input)
        
        # Refinar con los enhancers locales
        for n in range(self.n_local_enhancers):
            model_local = self.model[n+1]
            output_prev = model_local(input, output_prev)
            
        return output_prev

class GlobalGenerator(nn.Module):
    """Generador global base"""
    
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.InstanceNorm2d):
        super(GlobalGenerator, self).__init__()
        
        activation = nn.ReLU(True)
        
        # Capa inicial
        model = [nn.ReflectionPad2d(3), 
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), 
                 activation]
        
        # Capas de downsampling
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), 
                      activation]
        
        # Bloques residuales
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, activation=activation)]
        
        # Capas de upsampling
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), 
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)), 
                      activation]
        
        # Capa final
        model += [nn.ReflectionPad2d(3), 
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), 
                  nn.Tanh()]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, input):
        return self.model(input)

class LocalEnhancer(nn.Module):
    """Local enhancer para mejorar detalles de alta resolución"""
    
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_blocks_local=3, norm_layer=nn.InstanceNorm2d, n_local_enhancer=1):
        super(LocalEnhancer, self).__init__()
        
        self.n_local_enhancer = n_local_enhancer
        
        ###### Red global ###### 
        ngf_global = ngf * (2**n_local_enhancer)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, 
                                     n_blocks_global, norm_layer)
        
        ###### Red local ######        
        # Capas de downsampling
        model_downsample = [nn.ReflectionPad2d(3), 
                          nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                          norm_layer(ngf), 
                          nn.ReLU(True),
                          nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1), 
                          norm_layer(ngf * 2), 
                          nn.ReLU(True)]
        
        # Bloques residuales
        model_upsample = []
        for i in range(n_blocks_local):
            model_upsample += [ResnetBlock(ngf * 2, norm_layer=norm_layer, activation=nn.ReLU(True))]
        
        # Capas de upsampling
        model_upsample += [nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, 
                                            padding=1, output_padding=1), 
                          norm_layer(ngf), 
                          nn.ReLU(True), 
                          nn.ReflectionPad2d(3), 
                          nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), 
                          nn.Tanh()]
        
        self.model_global = model_global
        self.model_downsample = nn.Sequential(*model_downsample)
        self.model_upsample = nn.Sequential(*model_upsample)
        
    def forward(self, input, output_prev):
        # Crear versión de menor resolución de la entrada
        input_downsampled = F.interpolate(input, scale_factor=0.5**(self.n_local_enhancer+1), 
                                        mode='bilinear', align_corners=False)
        
        # Obtener salida del generador global en baja resolución
        output_prev_upsampled = F.interpolate(output_prev, scale_factor=2**(self.n_local_enhancer+1), 
                                            mode='bilinear', align_corners=False)
        
        # Procesar con red local
        hd = self.model_downsample(input)
        
        # Combinar features del generador global (upsampled) con features locales
        if self.n_local_enhancer == 1:
            # Para el primer enhancer, usar directamente la salida global
            hd = hd + F.interpolate(output_prev, size=hd.size()[2:], mode='bilinear', align_corners=False)
        
        output = self.model_upsample(hd)
        
        return output

class ResnetBlock(nn.Module):
    """Bloque residual con normalización e activación"""
    
    def __init__(self, dim, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0)]
        conv_block += [norm_layer(dim)]
        conv_block += [activation]
        
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
            
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0)]
        conv_block += [norm_layer(dim)]
        
        self.conv_block = nn.Sequential(*conv_block)
        
    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class MultiscaleDiscriminator(nn.Module):
    """Discriminador multiescala para Pix2PixHD"""
    
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        
        # Crear múltiples discriminadores
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            setattr(self, 'scale'+str(i), netD)
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        
    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]
            
    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        
        for i in range(num_D):
            model = getattr(self, 'scale'+str(i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
                
        return result

class NLayerDiscriminator(nn.Module):
    """Discriminador PatchGAN de N capas"""
    
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, 
                 use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        
        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
                     nn.LeakyReLU(0.2, True)]]
        
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                         norm_layer(nf), 
                         nn.LeakyReLU(0.2, True)]]
        
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                     norm_layer(nf),
                     nn.LeakyReLU(0.2, True)]]
        
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        
        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]
        
        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)
    
    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

# Funciones de pérdida
class GANLoss(nn.Module):
    """Pérdida GAN con soporte para diferentes tipos de pérdida"""
    
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
    
    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                           (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = torch.FloatTensor(input.size()).fill_(self.real_label)
                self.real_label_var = real_tensor.to(input.device)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                           (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = torch.FloatTensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = fake_tensor.to(input.device)
            target_tensor = self.fake_label_var
        return target_tensor
    
    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    """Pérdida perceptual usando VGG"""
    
    def __init__(self):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class Vgg19(nn.Module):
    """Red VGG19 para pérdida perceptual"""
    
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True).features
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

# Funciones de inicialización de pesos
def init_weights(net, init_type='normal', gain=0.02):
    """Inicializar pesos de la red"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Inicializar red con pesos y mover a GPU si está disponible"""
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net

# Función para crear el modelo
def create_model(input_nc=3, output_nc=1, ngf=64, ndf=64, n_layers_D=3, 
                n_downsample_global=4, n_blocks_global=9, n_local_enhancers=1, 
                n_blocks_local=3, norm='instance', gpu_ids=[], init_type='normal', init_gain=0.02):
    """Crear modelo Pix2PixHD completo"""
    
    if norm == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    
    # Crear generador
    netG = Generator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                    n_local_enhancers, n_blocks_local, norm_layer)
    
    # Crear discriminador
    netD = MultiscaleDiscriminator(input_nc + output_nc, ndf, n_layers_D, norm_layer, 
                                  use_sigmoid=False, num_D=3, getIntermFeat=True)
    
    # Inicializar redes
    netG = init_net(netG, init_type, init_gain, gpu_ids)
    netD = init_net(netD, init_type, init_gain, gpu_ids)
    
    return netG, netD

# Ejemplo de uso
if __name__ == "__main__":
    # Crear modelo para conversión RGB (3 canales) a NIR (1 canal)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parámetros del modelo
    input_nc = 3  # RGB
    output_nc = 1  # NIR
    ngf = 64
    ndf = 64
    
    # Crear generador y discriminador
    netG, netD = create_model(input_nc=input_nc, output_nc=output_nc, 
                             ngf=ngf, ndf=ndf, gpu_ids=[0] if torch.cuda.is_available() else [])
    
    print("Modelo Pix2PixHD creado exitosamente")
    print(f"Generador: {sum(p.numel() for p in netG.parameters())/1e6:.2f}M parámetros")
    print(f"Discriminador: {sum(p.numel() for p in netD.parameters())/1e6:.2f}M parámetros")
    
    # Prueba con tensor de ejemplo
    batch_size = 1
    height, width = 1024, 1024  # Ajustar según tu resolución objetivo
    
    # Crear tensor de entrada simulado
    input_tensor = torch.randn(batch_size, input_nc, height, width).to(device)
    
    # Forward pass del generador
    with torch.no_grad():
        output = netG(input_tensor)
        print(f"Forma de salida del generador: {output.shape}")
        
        # Forward pass del discriminador
        fake_pair = torch.cat([input_tensor, output], dim=1)
        disc_output = netD(fake_pair)
        print(f"Número de escalas del discriminador: {len(disc_output)}")
        for i, scale_output in enumerate(disc_output):
            print(f"Escala {i}: {len(scale_output)} features, última forma: {scale_output[-1].shape}")
