import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import torch.nn.functional as F

# Importar módulos personalizados
from model_architecture import Generator, MultiscaleDiscriminator
from loss_functions import GANLoss, VGGLoss, FeatureMatchingLossRobust
from Dataset import RGBNIRDataset, create_grid_image


class Pix2PixHDTrainer:
    def __init__(self, model_config_path, training_config_path):
        """Inicializar el entrenador Pix2PixHD con dos archivos de configuración"""
        # Cargar configuraciones por separado
        self.model_config = self.load_config(model_config_path, "modelo")
        self.training_config = self.load_config(training_config_path, "entrenamiento")

        # Combinar configuraciones en un solo diccionario para compatibilidad
        self.config = self.merge_configs(self.model_config, self.training_config)

        # Configurar dispositivo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")

        # Advertencia sobre uso de CPU
        if self.device.type == 'cpu':
            print("⚠️  ADVERTENCIA: Entrenando en CPU. Esto será muy lento.")
            print("   Considera usar GPU para mejor rendimiento.")

        # Configurar directorios
        self.setup_directories()

        # Inicializar modelos
        self.setup_models()

        # Configurar optimizadores
        self.setup_optimizers()

        # Configurar funciones de pérdida
        self.setup_losses()

        # Configurar dataset y dataloaders
        self.setup_data()

        # Configurar tensorboard
        self.writer = SummaryWriter(self.config['training']['log_dir'])

        # Variables de entrenamiento
        self.global_step = 0
        self.best_loss = float('inf')

        # Guardar configuración combinada para referencia
        self.save_merged_config()
        self.start_epoch = 1

    def load_config(self, config_path, config_type):
        """Cargar y validar archivo de configuración"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Archivo de configuración de {config_type} no encontrado: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✓ Configuración de {config_type} cargada desde: {config_path}")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error al parsear archivo YAML de {config_type}: {e}")

    def merge_configs(self, model_config, training_config):
        """Combinar las dos configuraciones en una estructura unificada"""
        merged_config = {
            'model': model_config.get('model', {}),
            'training': training_config.get('training', {}),
            'data': training_config.get('data', {}),
            'paths': training_config.get('paths', {})
        }

        # Validar que tenemos las secciones necesarias
        self.validate_merged_config(merged_config)

        return merged_config

    def validate_merged_config(self, config):
        """Validar que la configuración combinada tenga todos los campos necesarios"""
        required_sections = ['model', 'training', 'data']

        for section in required_sections:
            if section not in config:
                raise ValueError(f"Sección requerida '{section}' no encontrada en la configuración")

        # Validar campos específicos del modelo
        model_fields = [
            'input_channels', 'output_channels', 'generator', 'discriminator'
        ]
        for field in model_fields:
            if field not in config['model']:
                raise ValueError(f"Campo requerido 'model.{field}' no encontrado")

        # Validar campos específicos del entrenamiento
        training_fields = [
            'batch_size', 'lr', 'niter', 'checkpoint_dir', 'log_dir', 'sample_dir'
        ]
        for field in training_fields:
            if field not in config['training']:
                raise ValueError(f"Campo requerido 'training.{field}' no encontrado")

        # Validar campos de datos
        data_fields = ['train_rgb_dir', 'train_nir_dir', 'val_rgb_dir', 'val_nir_dir']
        for field in data_fields:
            if field not in config['data']:
                raise ValueError(f"Campo requerido 'data.{field}' no encontrado")

        print("✓ Validación de configuración completada")

    def save_merged_config(self):
        """Guardar la configuración combinada para referencia"""
        config_save_path = os.path.join(
            self.config['training']['checkpoint_dir'],
            'merged_config.yaml'
        )

        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

        print(f"Configuración combinada guardada en: {config_save_path}")

    def setup_directories(self):
        """Crear directorios necesarios"""
        dirs = [
            self.config['training']['checkpoint_dir'],
            self.config['training']['log_dir'],
            self.config['training']['sample_dir']
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def setup_models(self):
        """Inicializar modelos"""
        # Generador
        self.netG = Generator(
            input_nc=self.config['model']['input_channels'],
            output_nc=self.config['model']['output_channels'],
            ngf=self.config['model']['generator']['ngf'],
            n_downsample_global=self.config['model']['generator']['n_downsample_global'],
            n_blocks_global=self.config['model']['generator']['n_blocks_global'],
            n_local_enhancers=self.config['model']['generator']['n_local_enhancers'],
            n_blocks_local=self.config['model']['generator']['n_blocks_local']
        ).to(self.device)

        # Discriminador multiescala
        self.netD = MultiscaleDiscriminator(
            input_nc=self.config['model']['input_channels'] + self.config['model']['output_channels'],
            ndf=self.config['model']['discriminator']['ndf'],
            n_layers=self.config['model']['discriminator']['n_layers'],
            num_D=self.config['model']['discriminator']['num_D']
        ).to(self.device)

        print(f"Generador - Parámetros: {sum(p.numel() for p in self.netG.parameters()):,}")
        print(f"Discriminador - Parámetros: {sum(p.numel() for p in self.netD.parameters()):,}")

    def setup_optimizers(self):
        """Configurar optimizadores"""
        # Usar valores por defecto si no están definidos
        lr = self.config['training'].get('lr', 0.0002)
        beta1 = self.config['training'].get('beta1', 0.5)

        self.optimizer_G = optim.Adam(
            self.netG.parameters(),
            lr=lr,
            betas=(beta1, 0.999)
        )

        self.optimizer_D = optim.Adam(
            self.netD.parameters(),
            lr=lr,
            betas=(beta1, 0.999)
        )

        # Schedulers - usar valores por defecto si no están definidos
        niter_decay = self.config['training'].get('niter_decay', 100)

        self.scheduler_G = optim.lr_scheduler.LinearLR(
            self.optimizer_G,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=niter_decay
        )

        self.scheduler_D = optim.lr_scheduler.LinearLR(
            self.optimizer_D,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=niter_decay
        )

    def setup_losses(self):
        """Configurar funciones de pérdida"""
        self.criterionGAN = GANLoss(use_lsgan=True).to(self.device)
        self.criterionFeat = FeatureMatchingLossRobust(debug=False).to(self.device)
        self.criterionVGG = VGGLoss().to(self.device)
        self.criterionL1 = nn.L1Loss()

    def ensure_tensor_dimensions(self, tensor1, tensor2):
        """Asegurar que dos tensores tengan las mismas dimensiones espaciales"""
        if tensor1.shape[2:] != tensor2.shape[2:]:
            # Obtener las dimensiones mínimas comunes
            min_h = min(tensor1.shape[2], tensor2.shape[2])
            min_w = min(tensor1.shape[3], tensor2.shape[3])

            # Recortar ambos tensores a las dimensiones mínimas
            tensor1 = tensor1[:, :, :min_h, :min_w]
            tensor2 = tensor2[:, :, :min_h, :min_w]

        return tensor1, tensor2

    def setup_data(self):
        """Configurar dataset y dataloaders"""
        # Verificar que existan los directorios
        required_dirs = [
            self.config['data']['train_rgb_dir'],
            self.config['data']['train_nir_dir'],
            self.config['data']['val_rgb_dir'],
            self.config['data']['val_nir_dir']
        ]

        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directorio no encontrado: {dir_path}")

        # Configurar transformaciones
        # Usar dimensiones configurables o por defecto
        target_height = self.config['data'].get('image_height', 1944)
        target_width = self.config['data'].get('image_width', 2592)
        target_size = (target_height, target_width)

        # Transformación para entrenamiento con data augmentation
        train_transform_list = [
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]

        # Transformación para validación sin augmentation
        val_transform_list = [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]

        train_transform = transforms.Compose(train_transform_list)
        val_transform = transforms.Compose(val_transform_list)

        # Dataset de entrenamiento
        train_dataset = RGBNIRDataset(
            rgb_dir=self.config['data']['train_rgb_dir'],
            nir_dir=self.config['data']['train_nir_dir'],
            transform=train_transform,
            mode='train'
        )

        # Dataset de validación
        val_dataset = RGBNIRDataset(
            rgb_dir=self.config['data']['val_rgb_dir'],
            nir_dir=self.config['data']['val_nir_dir'],
            transform=val_transform,
            mode='val'
        )

        # DataLoaders con configuración optimizada
        num_workers = 0 if self.device.type == 'cpu' else self.config['training'].get('num_workers', 2)
        pin_memory = self.device.type == 'cuda'

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True  # Importante para training estable
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers
        )

        print(f"Dataset de entrenamiento: {len(train_dataset)} imágenes")
        print(f"Dataset de validación: {len(val_dataset)} imágenes")
        print(f"Tamaño de imagen objetivo: {target_size} (H x W)")

    def train_epoch(self, epoch):
        """Entrenar una época"""
        self.netG.train()
        self.netD.train()

        pbar = tqdm(self.train_loader, desc=f'Época {epoch}')
        epoch_losses = {'G': 0, 'D': 0, 'GAN': 0, 'Feat': 0, 'VGG': 0, 'L1': 0}

        for i, batch in enumerate(pbar):
            try:
                rgb_real = batch['rgb'].to(self.device)
                nir_real = batch['nir'].to(self.device)

                # Verificar y corregir dimensiones si es necesario
                rgb_real, nir_real = self.ensure_tensor_dimensions(rgb_real, nir_real)

                # Debug: imprimir dimensiones en la primera iteración
                if i == 0 and epoch == 1:
                    print(f"Dimensiones RGB: {rgb_real.shape}")
                    print(f"Dimensiones NIR: {nir_real.shape}")

                # Generar imagen NIR falsa
                nir_fake = self.netG(rgb_real)

                # Asegurar que nir_fake tenga las mismas dimensiones que nir_real
                nir_fake, nir_real = self.ensure_tensor_dimensions(nir_fake, nir_real)

                # Actualizar Discriminador
                self.optimizer_D.zero_grad()

                # Concatenar RGB y NIR para el discriminador
                real_input = torch.cat([rgb_real, nir_real], 1)
                fake_input = torch.cat([rgb_real, nir_fake.detach()], 1)

                # Asegurar que las entradas del discriminador tengan las mismas dimensiones
                real_input, fake_input = self.ensure_tensor_dimensions(real_input, fake_input)

                # Pérdida discriminador real
                pred_real = self.netD(real_input)
                loss_D_real = self.criterionGAN(pred_real, True)

                # Pérdida discriminador falso
                pred_fake = self.netD(fake_input)
                loss_D_fake = self.criterionGAN(pred_fake, False)

                # Pérdida total discriminador
                loss_D = (loss_D_real + loss_D_fake) * 0.5

                # Verificar que la pérdida sea válida
                if torch.isnan(loss_D) or torch.isinf(loss_D):
                    print(f"Pérdida D inválida en batch {i}, saltando...")
                    continue

                loss_D.backward()
                self.optimizer_D.step()

                # Actualizar Generador
                self.optimizer_G.zero_grad()

                # Re-generar predicción falsa para el generador (sin detach)
                fake_input_G = torch.cat([rgb_real, nir_fake], 1)
                pred_fake_G = self.netD(fake_input_G)

                # Pérdida GAN
                loss_G_GAN = self.criterionGAN(pred_fake_G, True)

                # Pérdida Feature Matching
                lambda_feat = self.config['training'].get('lambda_feat', 10.0)
                loss_G_Feat = self.criterionFeat(pred_fake_G, pred_real) * lambda_feat

                # Pérdida VGG
                lambda_vgg = self.config['training'].get('lambda_vgg', 10.0)
                loss_G_VGG = self.criterionVGG(nir_fake, nir_real) * lambda_vgg

                # Pérdida L1
                lambda_l1 = self.config['training'].get('lambda_l1', 100.0)
                loss_G_L1 = self.criterionL1(nir_fake, nir_real) * lambda_l1

                # Pérdida total generador
                loss_G = loss_G_GAN + loss_G_Feat + loss_G_VGG + loss_G_L1

                # Verificar que la pérdida sea válida
                if torch.isnan(loss_G) or torch.isinf(loss_G):
                    print(f"Pérdida G inválida en batch {i}, saltando...")
                    continue

                loss_G.backward()

                # Gradient clipping para estabilidad
                torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=10.0)

                self.optimizer_G.step()

                # Actualizar métricas
                epoch_losses['G'] += loss_G.item()
                epoch_losses['D'] += loss_D.item()
                epoch_losses['GAN'] += loss_G_GAN.item()
                epoch_losses['Feat'] += loss_G_Feat.item()
                epoch_losses['VGG'] += loss_G_VGG.item()
                epoch_losses['L1'] += loss_G_L1.item()

                # Actualizar barra de progreso
                pbar.set_postfix({
                    'G': f'{loss_G.item():.4f}',
                    'D': f'{loss_D.item():.4f}',
                    'GAN': f'{loss_G_GAN.item():.4f}',
                    'L1': f'{loss_G_L1.item():.4f}'
                })

                # Log a tensorboard
                print_freq = self.config['training'].get('print_freq', 100)
                if self.global_step % print_freq == 0:
                    self.writer.add_scalar('Loss/Generator', loss_G.item(), self.global_step)
                    self.writer.add_scalar('Loss/Discriminator', loss_D.item(), self.global_step)
                    self.writer.add_scalar('Loss/GAN', loss_G_GAN.item(), self.global_step)
                    self.writer.add_scalar('Loss/Feature_Matching', loss_G_Feat.item(), self.global_step)
                    self.writer.add_scalar('Loss/VGG', loss_G_VGG.item(), self.global_step)
                    self.writer.add_scalar('Loss/L1', loss_G_L1.item(), self.global_step)

                self.global_step += 1

            except Exception as e:
                print(f"Error en batch {i}: {e}")
                continue

        # Promediar pérdidas de la época
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def validate(self, epoch):
        """Validar el modelo"""
        self.netG.eval()
        val_loss = 0
        num_samples = 0

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                try:
                    rgb_real = batch['rgb'].to(self.device)
                    nir_real = batch['nir'].to(self.device)

                    # Asegurar dimensiones consistentes
                    rgb_real, nir_real = self.ensure_tensor_dimensions(rgb_real, nir_real)

                    nir_fake = self.netG(rgb_real)

                    # Asegurar que nir_fake tenga las mismas dimensiones que nir_real
                    nir_fake, nir_real = self.ensure_tensor_dimensions(nir_fake, nir_real)

                    loss = self.criterionL1(nir_fake, nir_real)
                    val_loss += loss.item()
                    num_samples += 1

                    # Guardar ejemplos de validación
                    save_freq = self.config['training'].get('save_freq', 10)
                    if i < 5 and epoch % save_freq == 0:
                        self.save_validation_images(rgb_real, nir_real, nir_fake, epoch, i)

                except Exception as e:
                    print(f"Error en validación batch {i}: {e}")
                    continue

        if num_samples > 0:
            val_loss /= num_samples
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        else:
            val_loss = float('inf')

        return val_loss

    def save_validation_images(self, rgb, nir_real, nir_fake, epoch, idx):
        """Guardar imágenes de validación"""
        try:
            # Desnormalizar imágenes
            rgb = (rgb + 1) / 2
            nir_real = (nir_real + 1) / 2
            nir_fake = (nir_fake + 1) / 2

            # Crear grid de imágenes
            grid = create_grid_image(rgb[0], nir_real[0], nir_fake[0])

            # Guardar imagen
            save_path = os.path.join(
                self.config['training']['sample_dir'],
                f'epoch_{epoch:03d}_sample_{idx:02d}.png'
            )
            grid.save(save_path)
        except Exception as e:
            print(f"Error guardando imagen de validación: {e}")

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Guardar checkpoint"""
        try:
            checkpoint = {
                'epoch': epoch,
                'global_step': self.global_step,
                'netG_state_dict': self.netG.state_dict(),
                'netD_state_dict': self.netD.state_dict(),
                'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                'scheduler_G_state_dict': self.scheduler_G.state_dict(),
                'scheduler_D_state_dict': self.scheduler_D.state_dict(),
                'val_loss': val_loss,
                'config': self.config
            }

            # Guardar checkpoint regular
            checkpoint_path = os.path.join(
                self.config['training']['checkpoint_dir'],
                f'checkpoint_epoch_{epoch:03d}.pth'
            )
            torch.save(checkpoint, checkpoint_path)

            # Guardar mejor modelo
            if is_best:
                best_path = os.path.join(
                    self.config['training']['checkpoint_dir'],
                    'best_model.pth'
                )
                torch.save(checkpoint, best_path)
                print(f"Nuevo mejor modelo guardado con val_loss: {val_loss:.6f}")

        except Exception as e:
            print(f"Error guardando checkpoint: {e}")

    def resume_training(self, checkpoint_path):
        """Reanudar entrenamiento desde checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.netG.load_state_dict(checkpoint['netG_state_dict'])
            self.netD.load_state_dict(checkpoint['netD_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
            self.global_step = checkpoint['global_step']
            self.best_loss = checkpoint.get('val_loss', float('inf'))
            return checkpoint['epoch']
        except Exception as e:
            print(f"Error cargando checkpoint: {e}")
            return 0

    def train(self):
        """Entrenamiento principal"""
        print("Iniciando entrenamiento...")
        start_time = time.time()

        niter = self.config['training']['niter']
        niter_fix_global = self.config['training'].get('niter_fix_global', 100)


        for epoch in range(self.start_epoch, niter + 1):
            # Entrenar época
            train_losses = self.train_epoch(epoch)

            # Validar
            val_loss = self.validate(epoch)

            # Actualizar schedulers después de cierto número de épocas
            if epoch > niter_fix_global:
                self.scheduler_G.step()
                self.scheduler_D.step()

            # Guardar checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss

            save_freq = self.config['training'].get('save_freq', 10)
            if epoch % save_freq == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)

            # Imprimir estadísticas
            elapsed = time.time() - start_time
            print(f"Época [{epoch}/{niter}] - "
                  f"Tiempo: {elapsed / 3600:.2f}h - "
                  f"G: {train_losses['G']:.4f} - "
                  f"D: {train_losses['D']:.4f} - "
                  f"Val: {val_loss:.6f}")

        print("Entrenamiento completado!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Entrenar Pix2PixHD para RGB a NIR')

    # Archivos de configuración separados (recomendado)
    parser.add_argument('--model-config', type=str, required=True,
                        help='Ruta al archivo de configuración del modelo (model_config.yml)')
    parser.add_argument('--training-config', type=str, required=True,
                        help='Ruta al archivo de configuración de entrenamiento (training_config.yml)')

    parser.add_argument('--resume', type=str, default=None,
                        help='Ruta al checkpoint para continuar entrenamiento')

    args = parser.parse_args()

    # Verificar que existan los archivos de configuración
    if not os.path.exists(args.model_config):
        print(f"Error: No se encontró el archivo de configuración del modelo {args.model_config}")
        sys.exit(1)

    if not os.path.exists(args.training_config):
        print(f"Error: No se encontró el archivo de configuración de entrenamiento {args.training_config}")
        sys.exit(1)

    try:
        # Inicializar entrenador
        trainer = Pix2PixHDTrainer(args.model_config, args.training_config)

        # Reanudar entrenamiento si se especifica
        start_epoch = 1
        if args.resume:
            if os.path.exists(args.resume):
                resumed_epoch = trainer.resume_training(args.resume)
                trainer.start_epoch = resumed_epoch + 1
                print(f"Reanudando entrenamiento desde época {trainer.start_epoch}")
            else:
                print(f"Warning: No se encontró el checkpoint {args.resume}")

        # Iniciar entrenamiento
        trainer.train()

    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()