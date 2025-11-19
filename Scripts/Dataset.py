import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class RGBNIRDataset(Dataset):
    """Dataset para imágenes RGB a NIR"""

    def __init__(self, rgb_dir, nir_dir, transform=None, mode='train'):
        """
        Args:
            rgb_dir (str): Directorio con imágenes RGB
            nir_dir (str): Directorio con imágenes NIR
            transform: Transformaciones a aplicar
            mode (str): 'train' o 'val'
        """
        self.rgb_dir = rgb_dir
        self.nir_dir = nir_dir
        # Siempre forzar tamaño fijo (preferiblemente divisible por 16)
        default_transform = transforms.Compose([
            transforms.CenterCrop((1944, 2592)),  # o Resize si es preferido
            transforms.ToTensor()
        ])

        self.transform = transform if transform is not None else default_transform
        self.mode = mode

        # Obtener lista de archivos RGB
        self.rgb_files = []
        if os.path.exists(rgb_dir):
            for file in os.listdir(rgb_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                    self.rgb_files.append(file)
            self.rgb_files.sort()

        # Verificar que existan las imágenes NIR correspondientes
        self.valid_pairs = []
        for rgb_file in self.rgb_files:
            # Buscar archivo NIR correspondiente (mismo nombre, posiblemente diferente extensión)
            base_name = os.path.splitext(rgb_file)[0]
            nir_file = None

            # Buscar con diferentes extensiones
            for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
                potential_nir = base_name + ext
                if os.path.exists(os.path.join(nir_dir, potential_nir)):
                    nir_file = potential_nir
                    break

            if nir_file:
                self.valid_pairs.append((rgb_file, nir_file))

        print(f"Dataset {mode}: {len(self.valid_pairs)} pares válidos encontrados")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        rgb_file, nir_file = self.valid_pairs[idx]

        # Cargar imágenes
        rgb_path = os.path.join(self.rgb_dir, rgb_file)
        nir_path = os.path.join(self.nir_dir, nir_file)

        try:
            rgb_image = Image.open(rgb_path).convert('RGB')
            nir_image = Image.open(nir_path).convert('RGB')  # Convertir NIR a RGB para compatibilidad

            # Aplicar transformaciones
            if self.transform:
                # Para entrenamiento, aplicar la misma transformación aleatoria a ambas imágenes
                if self.mode == 'train':
                    # Concatenar imágenes para aplicar la misma transformación
                    combined = Image.new('RGB', (rgb_image.width * 2, rgb_image.height))
                    combined.paste(rgb_image, (0, 0))
                    combined.paste(nir_image, (rgb_image.width, 0))

                    # Aplicar transformación
                    combined_transformed = self.transform(combined)

                    # Separar imágenes transformadas
                    _, h, w = combined_transformed.shape
                    w_half = w // 2
                    rgb_transformed = combined_transformed[:, :, :w_half]
                    nir_transformed = combined_transformed[:, :, w_half:]
                else:
                    # Para validación, aplicar transformación individualmente
                    rgb_transformed = self.transform(rgb_image)
                    nir_transformed = self.transform(nir_image)
            else:
                # Sin transformaciones, convertir a tensor
                to_tensor = transforms.ToTensor()
                rgb_transformed = to_tensor(rgb_image)
                nir_transformed = to_tensor(nir_image)

            return {
                'rgb': rgb_transformed,
                'nir': nir_transformed,
                'rgb_path': rgb_path,
                'nir_path': nir_path
            }

        except Exception as e:
            print(f"Error cargando imágenes {rgb_file}, {nir_file}: {e}")
            # Retornar tensor vacío en caso de error
            return {
                'rgb': torch.zeros(3, 1944, 2592),
                'nir': torch.zeros(3, 1944, 2592),
                'rgb_path': rgb_path,
                'nir_path': nir_path
            }


def save_checkpoint(model_state, optimizer_state, epoch, loss, filepath):
    """
    Guardar checkpoint del modelo

    Args:
        model_state: Estado del modelo (state_dict)
        optimizer_state: Estado del optimizador
        epoch: Época actual
        loss: Pérdida actual
        filepath: Ruta donde guardar el checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint guardado en: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """
    Cargar checkpoint del modelo

    Args:
        filepath: Ruta del checkpoint
        model: Modelo donde cargar los pesos
        optimizer: Optimizador (opcional)

    Returns:
        epoch, loss del checkpoint cargado
    """
    if not os.path.exists(filepath):
        print(f"No se encontró el checkpoint: {filepath}")
        return 0, float('inf')

    checkpoint = torch.load(filepath, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))

    print(f"Checkpoint cargado desde: {filepath}")
    print(f"Época: {epoch}, Pérdida: {loss}")

    return epoch, loss


def create_grid_image(rgb_tensor, nir_real_tensor, nir_fake_tensor):
    """
    Crear una imagen grid con RGB, NIR real y NIR generado

    Args:
        rgb_tensor: Tensor RGB (C, H, W)
        nir_real_tensor: Tensor NIR real (C, H, W)
        nir_fake_tensor: Tensor NIR generado (C, H, W)

    Returns:
        PIL Image con el grid
    """

    def tensor_to_pil(tensor):
        """Convertir tensor a PIL Image"""
        # Asegurar que el tensor esté en rango [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        # Convertir a numpy
        np_img = tensor.cpu().numpy().transpose(1, 2, 0)
        # Convertir a uint8
        np_img = (np_img * 255).astype(np.uint8)
        return Image.fromarray(np_img)

    # Convertir tensores a PIL Images
    rgb_pil = tensor_to_pil(rgb_tensor)
    nir_real_pil = tensor_to_pil(nir_real_tensor)
    nir_fake_pil = tensor_to_pil(nir_fake_tensor)

    # Obtener dimensiones
    width, height = rgb_pil.size

    # Crear imagen grid (3 imágenes horizontalmente)
    grid_width = width * 3
    grid_height = height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # Pegar imágenes
    grid_image.paste(rgb_pil, (0, 0))
    grid_image.paste(nir_real_pil, (width, 0))
    grid_image.paste(nir_fake_pil, (width * 2, 0))

    return grid_image