import albumentations as A
import cv2
import os
import glob
import numpy as np
from matplotlib import pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_FOLDER = "input-test"  
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# 2. DEFINE THE WEATHER PIPELINES
# ==========================================

weather_pipelines = {
    "rainy": A.Compose([
        A.RandomRain(
            brightness_coefficient=0.9, 
            drop_width=1, 
            blur_value=3, 
            p=1.0  
        ),
        A.MotionBlur(blur_limit=3, p=0.2) 
    ]),
    
    "snowy": A.Compose([
        A.RandomSnow(
            brightness_coeff=2.5, 
            snow_point_lower=0.1, 
            snow_point_upper=0.3, 
            p=1.0
        )
    ]),
    
    "foggy": A.Compose([
        A.RandomFog(
            fog_coef_lower=0.3, 
            fog_coef_upper=0.5, 
            alpha_coef=0.08, 
            p=1.0
        )
    ]),
    
    "sunny_glare": A.Compose([
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5), 
            angle_lower=0, 
            angle_upper=1, 
            num_flare_circles_lower=6, 
            num_flare_circles_upper=10, 
            src_radius=400, 
            src_color=(255, 255, 255), 
            p=1.0
        )
    ])
}

# ==========================================
# 3. THE GENERATOR ENGINE
# ==========================================
def process_images():

    image_paths = glob.glob(os.path.join(INPUT_FOLDER, "*"))
    print(f"Found {len(image_paths)} images in {INPUT_FOLDER}")

    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        for weather_type, pipeline in weather_pipelines.items():
            print(f"Applying {weather_type} to {filename}...")
            augmented = pipeline(image=image)['image']
            saved_image = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            
            output_filename = f"{name}_{weather_type}{ext}"
            save_path = os.path.join(OUTPUT_FOLDER, output_filename)
            cv2.imwrite(save_path, saved_image)

    print(f"\nSuccess! Augmented images saved to: {OUTPUT_FOLDER}")

# ==========================================
# 4. VISUALIZATION 
# ==========================================
def visualize_sample():
    images = glob.glob(os.path.join(OUTPUT_FOLDER, "*"))[:4] # Get first 4
    if not images: return

    plt.figure(figsize=(12, 6))
    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 4, i+1)
        plt.imshow(img)
        plt.title(os.path.basename(img_path))
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    process_images()
    visualize_sample()