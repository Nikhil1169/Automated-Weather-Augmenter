# Weather Augmentation Framework ⛈️🌨️☀️

A Python-based framework designed to generate synthetic weather data for computer vision applications. This tool allows researchers and developers to "stress test" AI models by algorithmically applying realistic rain, snow, fog, and sun glare to clean datasets.

## 📖 Overview
Machine learning models trained on clear images often fail when deployed in real-world weather conditions. This framework solves that problem by using the **Albumentations** library to create deterministic or probabilistic weather artifacts. It supports batch processing of large datasets (1000+ images) and is optimized for creating robust test sets.

## ✨ Features
* **Rain Generation:** Simulates rain streaks with motion blur.
* **Snow Simulation:** Adds snow particles and whitens the environment.
* **Fog/Haze:** Reduces contrast and adds atmospheric scattering.
* **Sun Glare:** Simulates lens flare and high-intensity light sources.
* **Batch Processing:** Automatically handles directory traversal and file I/O.
* **Color Safety:** Handles OpenCV BGR to RGB conversions automatically.
