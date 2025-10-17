import os
import requests
from pathlib import Path
import streamlit as st
from PIL import Image
import time

# Create test_images directory
TEST_IMAGES_DIR = Path("test_images")
TEST_IMAGES_DIR.mkdir(exist_ok=True)

# Sample image URLs for different use cases
SAMPLE_IMAGES = {
    "🔍 Real-time Object Detection": [
        "https://ultralytics.com/images/bus.jpg",
        "https://ultralytics.com/images/zidane.jpg",
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg"
    ],
    "🎨 Instance Segmentation": [
        "https://ultralytics.com/images/bus.jpg",
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg"
    ],
    "📊 Image Classification": [
        "https://ultralytics.com/images/bus.jpg",
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg"
    ],
    "🫴 Pose/Keypoint Estimation": [
        "https://ultralytics.com/images/bus.jpg",  # Note: need actual pose images
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg",
        "https://camo.githubusercontent.com/dd84236c640df6b479e47f26b51cce1b7a4cd8fb8/68747470733a2f2f6769746875622e636f6d2f756c7472616c79746963732f756c7472616c79746963732f6173736574732f7a6964616e652e6a7067"
    ],
    "📐 Oriented Object Detection (OBB)": [
        "https://ultralytics.com/images/bus.jpg"
    ],
    "🎥 Video Analytics": [
        "https://ultralytics.com/images/bus.jpg"
    ],
    "🚶 Multi-Object Tracking": [
        "https://ultralytics.com/images/bus.jpg",
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg"
    ],
    "👥 People Counting": [
        "https://ultralytics.com/images/bus.jpg",
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg"
    ],
    "⚠️ Zone Intrusion Alerts": [
        "https://ultralytics.com/images/bus.jpg"
    ],
    "🛡️ Safety Compliance (PPE)": [
        "https://ultralytics.com/images/bus.jpg"
    ],
    "🚗 Autonomous Driving": [
        "https://ultralytics.com/images/bus.jpg",
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg"
    ],
    "🛒 Retail Analytics": [
        "https://ultralytics.com/images/bus.jpg"
    ],
    "🏭 Manufacturing Inspection": [
        "https://ultralytics.com/images/bus.jpg"
    ],
    "📦 Logistics & Warehouse": [
        "https://ultralytics.com/images/bus.jpg"
    ],
    "🏗️ Construction Safety": [
        "https://ultralytics.com/images/bus.jpg"
    ],
    "🔒 Security & Surveillance": [
        "https://ultralytics.com/images/bus.jpg",
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg"
    ],
    "🏥 Healthcare Imaging": [
        "https://ultralytics.com/images/bus.jpg"
    ],
    "🌾 Agriculture Monitoring": [
        "https://ultralytics.com/images/bus.jpg"
    ],
    "🎮 Sports Analytics": [
        "https://ultralytics.com/images/bus.jpg"
    ],
    "📄 Document/Barcode Detection": [
        "https://ultralytics.com/images/bus.jpg"
    ],
    "🚖 License Plate Recognition": [
        "https://ultralytics.com/images/bus.jpg"
    ],
    "🔥 Fire/Smoke Detection": [
        "https://ultralytics.com/images/bus.jpg"
    ],
    "📄 Document Analysis (Text Detection)": [
        "https://ultralytics.com/images/bus.jpg"
    ]
}

# Additional diverse test images
ADDITIONAL_TEST_IMAGES = [
    "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=640",  # Vehicles on road
    "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=640",  # Construction site
    "https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=640",  # People in store
    "https://images.unsplash.com/photo-1611273426858-450d8e3c9fce?w=640",  # Warehouse
    "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=640",  # Sports field
    "https://images.unsplash.com/photo-1621905252475-8e9c075d6e17?w=640",  # License plate
    "https://images.unsplash.com/photo-1544568100-847a948585b9?w=640",  # Security camera
    "https://images.unsplash.com/photo-1594736797933-d0401ba2fe65?w=640",  # Agriculture
    "https://images.unsplash.com/photo-1576671081837-49000212a370?w=640",  # Medical
    "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=640",  # Document
]

def download_test_images(progress_bar=None, status_text=None):
    """Download test images for different use cases"""
    total_images = sum(len(urls) for urls in SAMPLE_IMAGES.values()) + len(ADDITIONAL_TEST_IMAGES)
    downloaded = 0

    if progress_bar:
        progress_bar.progress(0, text="Starting download...")

    # Download use-case specific images
    for use_case, urls in SAMPLE_IMAGES.items():
        # Sanitize directory name - remove emojis and special chars
        clean_name = ''.join(c for c in use_case if c.isalnum() or c in (' ', '-', '_')).rstrip()
        use_case_dir = TEST_IMAGES_DIR / clean_name.replace(" ", "_").replace("-", "_")
        use_case_dir.mkdir(exist_ok=True)

        for i, url in enumerate(urls):
            try:
                filename = f"sample_{i+1}_{Path(url).name or 'image.jpg'}"
                filepath = use_case_dir / filename

                if not filepath.exists():
                    response = requests.get(url, timeout=30, stream=True)
                    response.raise_for_status()

                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                downloaded += 1
                if progress_bar:
                    progress = downloaded / total_images
                    progress_bar.progress(progress, text=f"Downloaded {downloaded}/{total_images} images")
                    if status_text:
                        status_text.text(f"Saved to: {filepath}")

            except Exception as e:
                if status_text:
                    status_text.text(f"Failed to download {url}: {str(e)}")
                continue

    # Download additional diverse images
    diverse_dir = TEST_IMAGES_DIR / "diverse_scenarios"
    diverse_dir.mkdir(exist_ok=True)

    for i, url in enumerate(ADDITIONAL_TEST_IMAGES):
        try:
            filename = f"diverse_{i+1}_{Path(url).name or 'image.jpg'}"
            filepath = diverse_dir / filename

            if not filepath.exists():
                response = requests.get(url, timeout=30, stream=True)
                response.raise_for_status()

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            downloaded += 1
            if progress_bar:
                progress = downloaded / total_images
                progress_bar.progress(progress, text=f"Downloaded {downloaded}/{total_images} images")
                if status_text:
                    status_text.text(f"Saved to: {filepath}")

        except Exception as e:
            if status_text:
                status_text.text(f"Failed to download {url}: {str(e)}")
            continue

    if progress_bar:
        progress_bar.progress(1.0, text="Download complete!")
        if status_text:
            status_text.text(f"✅ Successfully downloaded {downloaded} test images to {TEST_IMAGES_DIR}")

    return downloaded

def get_local_test_images(use_case=None):
    """Get list of local test images for a use case"""
    if use_case:
        # Use the same sanitization logic as in download function
        clean_name = ''.join(c for c in use_case if c.isalnum() or c in (' ', '-', '_')).rstrip()
        use_case_dir = TEST_IMAGES_DIR / clean_name.replace(" ", "_").replace("-", "_")
        if use_case_dir.exists():
            return [str(f) for f in use_case_dir.glob("*.jpg")] + [str(f) for f in use_case_dir.glob("*.png")]
    else:
        # Get all test images
        all_images = []
        for use_case_dir in TEST_IMAGES_DIR.glob("*"):
            if use_case_dir.is_dir():
                all_images.extend([str(f) for f in use_case_dir.glob("*.jpg")])
                all_images.extend([str(f) for f in use_case_dir.glob("*.png")])
        return all_images

    return []

def create_sample_datasets_info():
    """Create information about available datasets"""
    dataset_info = {
        "COCO": {
            "description": "Microsoft Common Objects in Context - 80 object classes",
            "classes": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
            "url": "https://cocodataset.org/",
            "tasks": ["detect", "segment", "pose"]
        },
        "ImageNet": {
            "description": "ImageNet Large Scale Visual Recognition Challenge - 1000 classes",
            "classes": ["See ImageNet website for complete class list"],
            "url": "https://www.image-net.org/",
            "tasks": ["classify"]
        },
        "DOTA": {
            "description": "Dataset for Object Detection in Aerial Images - Oriented bounding boxes",
            "classes": ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle", "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool"],
            "url": "https://captain-whu.github.io/DOTA/",
            "tasks": ["obb"]
        },
        "BDD100K": {
            "description": "Berkeley Deep Drive - Autonomous driving dataset with videos and images",
            "classes": ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"],
            "url": "https://bdd-data.berkeley.edu/",
            "tasks": ["detect", "segment", "track"]
        },
        "COCO-Text": {
            "description": "Scene text detection and recognition on COCO images",
            "classes": ["text"],
            "url": "https://bgshih.github.io/cocotext/",
            "tasks": ["detect"]
        }
    }
    return dataset_info

def display_dataset_info():
    """Display information about supported datasets in Streamlit"""
    datasets = create_sample_datasets_info()

    st.header("📊 Supported Datasets")

    for name, info in datasets.items():
        with st.expander(f"{name} - {info['description'][:50]}..."):
            st.write(f"**Full Description:** {info['description']}")
            st.write(f"**Sample Classes:** {', '.join(info['classes'][:10])}{'...' if len(info['classes']) > 10 else ''}")
            st.write(f"**Supported Tasks:** {', '.join(info['tasks'])}")
            st.write(f"**[Dataset Link]({info['url']})**")

if __name__ == "__main__":
    # Run download when executed directly
    st.info("Downloading test images...")
    downloaded = download_test_images()
    st.success(f"Downloaded {downloaded} test images!")
    st.info(f"Images saved to: {TEST_IMAGES_DIR}")
