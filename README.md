# RGB2NIR-Pineapple-Pix2PixHD
Implementation code for the Pix2PixHD model used for RGB-to-Synthetic NIR image translation. The goal is to facilitate pineapple crop monitoring and early detection of crop stress.

# RGB to Synthetic NIR: Image Translation for Pineapple Crop Monitoring

This repository contains the training and evaluation code for the image-to-image translation model used in our research on precision agriculture.

## 🎯 Project Goal

The primary objective is to generate **synthetic Near-Infrared (NIR)** images from standard Red-Green-Blue (RGB) aerial imagery of pineapple crops. This process is crucial for facilitating early stress detection, health monitoring, and advanced crop analysis where dedicated NIR sensors are unavailable or costly.

## 🧠 Model Architecture

The core of this project is based on the **Pix2PixHD** Generative Adversarial Network (GAN) architecture, adapted for generating high-resolution (HD) images for spectral band translation.

## ⚙️ Training

To train the model, ensure your dataset is organized in paired RGB and NIR folders as required by the Pix2PixHD framework. The images used for this project will be available soon.
