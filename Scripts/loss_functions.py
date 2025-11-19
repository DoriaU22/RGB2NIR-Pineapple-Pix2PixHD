import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import namedtuple


class GANLoss(nn.Module):
    """Pérdida GAN estándar con soporte para LSGAN"""

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):
        """Crear tensor objetivo con el mismo tamaño que la predicción"""
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        """
        Calcular pérdida GAN
        Args:
            prediction: lista de tensores o tensor único de predicciones del discriminador
            target_is_real: bool, si el objetivo es real o falso
        """
        if isinstance(prediction, list):
            # Para discriminador multiescala
            loss = 0
            for pred_i in prediction:
                if isinstance(pred_i, list):
                    # Tomar la última predicción de cada escala
                    pred_i = pred_i[-1]
                target_tensor = self.get_target_tensor(pred_i, target_is_real)
                loss += self.loss(pred_i, target_tensor)
            return loss
        else:
            # Para discriminador único
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target_tensor)


class FeatureMatchingLoss(nn.Module):
    """Pérdida de emparejamiento de características (Feature Matching Loss)"""

    def __init__(self):
        super(FeatureMatchingLoss, self).__init__()
        self.criterionFeat = nn.L1Loss()

    def forward(self, pred_fake, pred_real):
        """
        Calcular pérdida de emparejamiento de características
        Args:
            pred_fake: predicciones del discriminador para imágenes falsas
            pred_real: predicciones del discriminador para imágenes reales
        """
        loss_G_Feat = 0

        # Manejar diferentes estructuras de salida del discriminador
        if not isinstance(pred_fake, list):
            pred_fake = [pred_fake]
        if not isinstance(pred_real, list):
            pred_real = [pred_real]

        # Para cada escala del discriminador
        for i in range(len(pred_fake)):
            # Verificar que ambas predicciones tengan la misma longitud
            if i >= len(pred_real):
                break

            # Obtener características de cada escala
            if isinstance(pred_fake[i], list) and isinstance(pred_real[i], list):
                # Discriminador devuelve lista de características
                fake_feats = pred_fake[i]
                real_feats = pred_real[i]

                # Asegurar que ambas listas tengan el mismo tamaño
                min_len = min(len(fake_feats), len(real_feats))

                # Comparar características intermedias (no la predicción final)
                for j in range(min_len - 1):  # -1 para excluir la predicción final
                    if fake_feats[j].numel() > 0 and real_feats[j].numel() > 0:
                        # Asegurar que las características tengan las mismas dimensiones
                        if fake_feats[j].shape == real_feats[j].shape:
                            loss_G_Feat += self.criterionFeat(fake_feats[j], real_feats[j])
                        else:
                            # Redimensionar si es necesario
                            min_size = [min(fake_feats[j].shape[k], real_feats[j].shape[k])
                                        for k in range(len(fake_feats[j].shape))]
                            fake_resized = fake_feats[j][:min_size[0], :min_size[1]] if len(min_size) >= 2 else \
                            fake_feats[j]
                            real_resized = real_feats[j][:min_size[0], :min_size[1]] if len(min_size) >= 2 else \
                            real_feats[j]
                            loss_G_Feat += self.criterionFeat(fake_resized, real_resized)
            else:
                # Discriminador devuelve tensor único
                if pred_fake[i].shape == pred_real[i].shape:
                    loss_G_Feat += self.criterionFeat(pred_fake[i], pred_real[i])

        return loss_G_Feat


class VGGLoss(nn.Module):
    """Pérdida perceptual usando características de VGG"""

    def __init__(self, layids=None):
        super(VGGLoss, self).__init__()
        if layids is None:
            layids = [4, 9, 16, 23]  # Conv2_2, Conv3_2, Conv4_2, Conv5_2

        self.vgg = Vgg16()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        """
        Calcular pérdida VGG
        Args:
            x: imagen generada
            y: imagen objetivo
        """
        # Convertir imágenes de un canal a tres canales si es necesario
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.size(1) == 1:
            y = y.repeat(1, 3, 1, 1)

        # Redimensionar si las imágenes son muy grandes para VGG
        if x.size(2) > 512 or x.size(3) > 512:
            x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(512, 512), mode='bilinear', align_corners=False)

        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)

        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss


class Vgg16(nn.Module):
    """Red VGG16 para extracción de características"""

    def __init__(self):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        # Congelar parámetros de VGG
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)

        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class PerceptualLoss(nn.Module):
    """Pérdida perceptual combinada"""

    def __init__(self, use_vgg=True, use_feat_match=True, use_gan=True):
        super(PerceptualLoss, self).__init__()

        self.use_vgg = use_vgg
        self.use_feat_match = use_feat_match
        self.use_gan = use_gan

        if use_vgg:
            self.vgg_loss = VGGLoss()
        if use_feat_match:
            self.feat_loss = FeatureMatchingLoss()
        if use_gan:
            self.gan_loss = GANLoss(use_lsgan=True)

    def forward(self, fake_img, real_img, pred_fake=None, pred_real=None, target_is_real=True):
        """
        Calcular pérdida perceptual combinada
        """
        total_loss = 0
        losses = {}

        if self.use_vgg:
            vgg_loss = self.vgg_loss(fake_img, real_img)
            total_loss += vgg_loss
            losses['vgg'] = vgg_loss

        if self.use_feat_match and pred_fake is not None and pred_real is not None:
            feat_loss = self.feat_loss(pred_fake, pred_real)
            total_loss += feat_loss
            losses['feat'] = feat_loss

        if self.use_gan and pred_fake is not None:
            gan_loss = self.gan_loss(pred_fake, target_is_real)
            total_loss += gan_loss
            losses['gan'] = gan_loss

        return total_loss, losses


# Función auxiliar para debugging
def print_discriminator_output_structure(pred, name=""):
    """Función auxiliar para debuggear la estructura de salida del discriminador"""
    print(f"\n=== Estructura de {name} ===")
    if isinstance(pred, list):
        print(f"Lista con {len(pred)} elementos:")
        for i, item in enumerate(pred):
            if isinstance(item, list):
                print(f"  [{i}]: Lista con {len(item)} elementos")
                for j, subitem in enumerate(item):
                    if hasattr(subitem, 'shape'):
                        print(f"    [{j}]: Tensor shape {subitem.shape}")
                    else:
                        print(f"    [{j}]: {type(subitem)}")
            elif hasattr(item, 'shape'):
                print(f"  [{i}]: Tensor shape {item.shape}")
            else:
                print(f"  [{i}]: {type(item)}")
    elif hasattr(pred, 'shape'):
        print(f"Tensor shape: {pred.shape}")
    else:
        print(f"Tipo: {type(pred)}")
    print("=" * 30)


# Versión robusta de FeatureMatchingLoss con debugging - CORREGIDA
class FeatureMatchingLossRobust(nn.Module):
    """Versión robusta de FeatureMatchingLoss con manejo de errores y debugging"""

    def __init__(self, debug=False):
        super(FeatureMatchingLossRobust, self).__init__()
        self.criterionFeat = nn.L1Loss()
        self.debug = debug

    def forward(self, pred_fake, pred_real):
        """Calcular pérdida de emparejamiento de características con manejo robusto de errores"""

        if self.debug:
            print_discriminator_output_structure(pred_fake, "pred_fake")
            print_discriminator_output_structure(pred_real, "pred_real")

        # Determinar el dispositivo correctamente
        device = None
        if isinstance(pred_fake, list) and len(pred_fake) > 0:
            if isinstance(pred_fake[0], list) and len(pred_fake[0]) > 0:
                device = pred_fake[0][0].device
            elif hasattr(pred_fake[0], 'device'):
                device = pred_fake[0].device
        elif hasattr(pred_fake, 'device'):
            device = pred_fake.device

        if device is None:
            device = torch.device('cpu')

        # Inicializar loss como tensor con gradiente
        loss_G_Feat = torch.tensor(0.0, device=device, requires_grad=True)
        feat_count = 0

        try:
            # Normalizar entradas a listas
            if not isinstance(pred_fake, list):
                pred_fake = [pred_fake]
            if not isinstance(pred_real, list):
                pred_real = [pred_real]

            # Verificar que ambas listas tengan elementos
            if len(pred_fake) == 0 or len(pred_real) == 0:
                if self.debug:
                    print("Warning: Una de las predicciones está vacía")
                return loss_G_Feat

            # Procesar cada escala
            num_scales = min(len(pred_fake), len(pred_real))

            for i in range(num_scales):
                try:
                    fake_i = pred_fake[i]
                    real_i = pred_real[i]

                    # Caso 1: Ambos son listas (características múltiples por escala)
                    if isinstance(fake_i, list) and isinstance(real_i, list):
                        num_feats = min(len(fake_i), len(real_i))

                        # Excluir la predicción final (última característica)
                        for j in range(max(0, num_feats - 1)):
                            try:
                                fake_feat = fake_i[j]
                                real_feat = real_i[j]

                                # Verificar que sean tensores válidos
                                if (hasattr(fake_feat, 'shape') and hasattr(real_feat, 'shape') and
                                        fake_feat.numel() > 0 and real_feat.numel() > 0):

                                    # Ajustar dimensiones si es necesario
                                    if fake_feat.shape != real_feat.shape:
                                        min_dims = [min(fake_feat.shape[k], real_feat.shape[k])
                                                    for k in range(len(fake_feat.shape))]

                                        # Recortar a dimensiones mínimas
                                        slices = tuple(slice(0, dim) for dim in min_dims)
                                        fake_feat = fake_feat[slices]
                                        real_feat = real_feat[slices]

                                    # IMPORTANTE: Usar suma en lugar de asignación para mantener el grafo
                                    feat_loss = self.criterionFeat(fake_feat, real_feat)
                                    loss_G_Feat = loss_G_Feat + feat_loss
                                    feat_count += 1

                            except Exception as e:
                                if self.debug:
                                    print(f"Error procesando característica [{i}][{j}]: {e}")
                                continue

                    # Caso 2: Ambos son tensores directamente
                    elif (hasattr(fake_i, 'shape') and hasattr(real_i, 'shape')):
                        if fake_i.numel() > 0 and real_i.numel() > 0:
                            if fake_i.shape == real_i.shape:
                                feat_loss = self.criterionFeat(fake_i, real_i)
                                loss_G_Feat = loss_G_Feat + feat_loss
                                feat_count += 1
                            elif self.debug:
                                print(f"Shapes no coinciden en escala {i}: {fake_i.shape} vs {real_i.shape}")

                except Exception as e:
                    if self.debug:
                        print(f"Error procesando escala {i}: {e}")
                    continue

            # Normalizar por el número de características procesadas
            if feat_count > 0:
                loss_G_Feat = loss_G_Feat / feat_count

            if self.debug:
                print(f"FeatureMatchingLoss: {loss_G_Feat.item():.6f} (de {feat_count} características)")

            return loss_G_Feat

        except Exception as e:
            print(f"Error crítico en FeatureMatchingLoss: {e}")
            # Retornar pérdida cero como tensor válido
            return torch.tensor(0.0, device=device, requires_grad=True)