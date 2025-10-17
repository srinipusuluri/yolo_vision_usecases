import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
import io
import tempfile
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from utils import download_test_images, get_local_test_images, display_dataset_info
import json
import time

# Set page config
st.set_page_config(
    page_title="YOLO Vision Models - All Use Cases",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🤖 YOLO Vision Models - Comprehensive Use Cases Platform")
st.markdown("Explore YOLO's capabilities across object detection, segmentation, classification, pose estimation, OBB, tracking, and more!")

# Sidebar menu for use cases
st.sidebar.header("🎯 Select Use Case")

use_cases = [
    "🔍 Real-time Object Detection",
    "🎨 Instance Segmentation",
    "📊 Image Classification",
    "🫴 Pose/Keypoint Estimation",
    "📐 Oriented Object Detection (OBB)",
    "🎥 Video Analytics",
    "🚶 Multi-Object Tracking",
    "👥 People Counting",
    "⚠️ Zone Intrusion Alerts",
    "🛡️ Safety Compliance (PPE)",
    "🚗 Autonomous Driving",
    "🛒 Retail Analytics",
    "🏭 Manufacturing Inspection",
    "📦 Logistics & Warehouse",
    "🏗️ Construction Safety",
    "🔒 Security & Surveillance",
    "🏥 Healthcare Imaging",
    "🌾 Agriculture Monitoring",
    "🎮 Sports Analytics",
    "📄 Document/Barcode Detection",
    "🚖 License Plate Recognition",
    "🔥 Fire/Smoke Detection",
    "📄 Document Analysis (Text Detection)"
]

selected_case = st.sidebar.selectbox("Choose a use case:", use_cases)

# Model selection
st.sidebar.header("⚙️ Model Configuration")
model_sizes = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
selected_model = st.sidebar.selectbox("YOLO Model Size:", model_sizes, index=1)

# Task mapping
task_mapping = {
    "🔍 Real-time Object Detection": ("detect", "coco"),
    "🎨 Instance Segmentation": ("segment", "coco"),
    "📊 Image Classification": ("classify", "imagenet"),
    "🫴 Pose/Keypoint Estimation": ("pose", "coco"),
    "📐 Oriented Object Detection (OBB)": ("obb", "dota8"),
    "🎥 Video Analytics": ("detect", "coco"),
    "🚶 Multi-Object Tracking": ("track", "coco"),
    "👥 People Counting": ("detect", "coco"),
    "⚠️ Zone Intrusion Alerts": ("detect", "coco"),
    "🛡️ Safety Compliance (PPE)": ("detect", "coco"),
    "🚗 Autonomous Driving": ("detect", "coco"),
    "🛒 Retail Analytics": ("detect", "coco"),
    "🏭 Manufacturing Inspection": ("detect", "coco"),
    "📦 Logistics & Warehouse": ("detect", "coco"),
    "🏗️ Construction Safety": ("detect", "coco"),
    "🔒 Security & Surveillance": ("detect", "coco"),
    "🏥 Healthcare Imaging": ("detect", "coco"),  # Would need medical data
    "🌾 Agriculture Monitoring": ("detect", "coco"),
    "🎮 Sports Analytics": ("detect", "coco"),
    "📄 Document/Barcode Detection": ("detect", "coco"),
    "🚖 License Plate Recognition": ("detect", "coco"),
    "🔥 Fire/Smoke Detection": ("detect", "coco"),
    "📄 Document Analysis (Text Detection)": ("detect", "coco")
}

# Confidence threshold
confidence = st.sidebar.slider("Confidence Threshold:", 0.1, 1.0, 0.25)

# Test Images Download
st.sidebar.header("📥 Test Images")
if st.sidebar.button("📥 Download Test Images", help="Download sample images for all use cases"):
    progress_placeholder = st.sidebar.empty()
    status_placeholder = st.sidebar.empty()
    progress_bar = progress_placeholder.progress(0, text="Preparing download...")
    status_text = status_placeholder.empty()

    try:
        downloaded = download_test_images(progress_bar, status_text)
        progress_bar.empty()
        status_placeholder.success(f"✅ Downloaded {downloaded} test images!")
    except Exception as e:
        progress_bar.empty()
        status_placeholder.error(f"❌ Download failed: {str(e)}")

# Input options
st.sidebar.header("📥 Input Source")
input_option = st.sidebar.radio("Select input type:", ["Upload Image", "Camera", "URL", "Sample Images", "Local Test Images", "Video Upload"])

# Sample images for testing
sample_images = {
    "🔍 Real-time Object Detection": "https://ultralytics.com/images/bus.jpg",
    "🎨 Instance Segmentation": "https://ultralytics.com/images/bus.jpg",
    "📊 Image Classification": "https://ultralytics.com/images/bus.jpg",
    "🫴 Pose/Keypoint Estimation": "https://ultralytics.com/images/bus.jpg",  # Need people image
    "📐 Oriented Object Detection (OBB)": "https://ultralytics.com/images/bus.jpg",
    "🚶 Multi-Object Tracking": "https://ultralytics.com/images/bus.jpg",
    "👥 People Counting": "https://ultralytics.com/images/bus.jpg",
    "⚠️ Zone Intrusion Alerts": "https://ultralytics.com/images/bus.jpg",
    "🛡️ Safety Compliance (PPE)": "https://ultralytics.com/images/bus.jpg",
    "🚗 Autonomous Driving": "https://ultralytics.com/images/bus.jpg",
    "🛒 Retail Analytics": "https://ultralytics.com/images/bus.jpg",
    "🏭 Manufacturing Inspection": "https://ultralytics.com/images/bus.jpg",
    "📦 Logistics & Warehouse": "https://ultralytics.com/images/bus.jpg",
    "🏗️ Construction Safety": "https://ultralytics.com/images/bus.jpg",
    "🔒 Security & Surveillance": "https://ultralytics.com/images/bus.jpg",
    "🏥 Healthcare Imaging": "https://ultralytics.com/images/bus.jpg",
    "🌾 Agriculture Monitoring": "https://ultralytics.com/images/bus.jpg",
    "🎮 Sports Analytics": "https://ultralytics.com/images/bus.jpg",
    "📄 Document/Barcode Detection": "https://ultralytics.com/images/bus.jpg",
    "🚖 License Plate Recognition": "https://ultralytics.com/images/bus.jpg",
    "🔥 Fire/Smoke Detection": "https://ultralytics.com/images/bus.jpg",
    "📄 Document Analysis (Text Detection)": "https://ultralytics.com/images/bus.jpg"
}

# Load model function
@st.cache_resource
def load_model(model_name, task):
    try:
        model = YOLO(f"{model_name}-{task}.pt")
        return model
    except:
        st.error(f"Failed to load model {model_name}-{task}.pt. Using default detection model.")
        return YOLO("yolov8n.pt")

def query_ollama(prompt, model="llama3.2:3b", max_tokens=300):
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens}
        }, timeout=30)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: {response.status_code} {response.text}"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}. Make sure Ollama is running locally."

def create_detections_summary(result_df, use_case):
    if not result_df:
        return "No objects detected."
    summary = f"Detected objects in this {use_case.lower()}:\n"
    summary += "\n".join([f"- {row['Class']} with confidence {row['Confidence']} at coordinates ({row['X1']}, {row['Y1']}, {row['X2']}, {row['Y2']})" for row in result_df[:10]])  # Limit to 10 detections
    if len(result_df) > 10:
        summary += f"\n... and {len(result_df) - 10} more detections"
    return summary

# Main content area
st.header("🤖 Ask AI about YOLO and Vision Models")
general_question = st.text_input("Enter your question about YOLO models, use cases, or computer vision:",
                              key=f"general_question_{time.time()}")
if st.button("🤖 Ask AI", key=f"general_ask_{time.time()}") and general_question:
    with st.spinner("Generating AI response..."):
        system_prompt = f"You are an expert AI assistant specializing in computer vision and YOLO models. Answer questions about YOLO (You Only Look Once) vision models, their capabilities, use cases, and computer vision concepts. If the question is about a specific use case from this platform, consider the context of the selected use case '{selected_case}'.\n\nQuestion: '{general_question}'\n\nProvide a helpful, accurate, and detailed response."
        ai_response = query_ollama(system_prompt)
    st.markdown(f"**🤖 AI Response:**\n\n{ai_response}")

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.header(f"{selected_case}")

    # Description based on use case
    descriptions = {
        "🔍 Real-time Object Detection": "Detect and locate objects in real-time with bounding boxes.",
        "🎨 Instance Segmentation": "Segment individual instances of objects with pixel-level masks.",
        "📊 Image Classification": "Classify the main content of an image into categories.",
        "🫴 Pose/Keypoint Estimation": "Estimate human pose keypoints for body joint detection.",
        "📐 Oriented Object Detection (OBB)": "Detect objects with oriented bounding boxes for aerial/drone imagery.",
        "🎥 Video Analytics": "Process video frames for continuous object detection.",
        "🚶 Multi-Object Tracking": "Track objects across frames with identity preservation.",
        "👥 People Counting": "Count people in crowded scenes with zone-based counting.",
        "⚠️ Zone Intrusion Alerts": "Alert when objects enter predefined zones.",
        "🛡️ Safety Compliance (PPE)": "Detect safety equipment like helmets and vests.",
        "🚗 Autonomous Driving": "Detect vehicles, pedestrians, and traffic signals.",
        "🛒 Retail Analytics": "Monitor shelves, products, and customer behavior.",
        "🏭 Manufacturing Inspection": "Quality control for defects and assembly verification.",
        "📦 Logistics & Warehouse": "Track packages, pallets, and inventory.",
        "🏗️ Construction Safety": "Monitor PPE compliance and hazard detection.",
        "🔒 Security & Surveillance": "Detect intrusions, abandoned objects, and activities.",
        "🏥 Healthcare Imaging": "Analyze medical images for diagnosis assistance.",
        "🌾 Agriculture Monitoring": "Detect crops, weeds, and livestock.",
        "🎮 Sports Analytics": "Track players, balls, and analyze gameplay.",
        "📄 Document/Barcode Detection": "Extract text and codes from documents.",
        "🚖 License Plate Recognition": "OCR vehicle license plates.",
        "🔥 Fire/Smoke Detection": "Early detection of fire hazards.",
        "📄 Document Analysis (Text Detection)": "Detect and extract text from scene images."
    }

    st.markdown(f"**Description:** {descriptions.get(selected_case, 'General YOLO vision task')}")

    # Load input based on option
    image = None
    user_instructions = None
    video_file = None

    if input_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp", "tiff"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            user_instructions = st.text_area("Enter your instructions for what you need from this picture:",
                                           placeholder="e.g., Describe the scene, count the people, what objects are visible?",
                                           key="user_instructions_upload")

    elif input_option == "Camera":
        camera_input = st.camera_input("Take a picture")
        if camera_input is not None:
            image = Image.open(camera_input)
            st.image(image, caption="Camera Input", use_column_width=True)
            user_instructions = st.text_area("Enter your instructions for what you need from this picture:",
                                           placeholder="e.g., Describe the scene, count the people, what objects are visible?",
                                           key="user_instructions_camera")

    elif input_option == "URL":
        url = st.text_input("Enter image URL:")
        if url:
            try:
                response = requests.get(url, timeout=10)
                image = Image.open(io.BytesIO(response.content))
                st.image(image, caption="Image from URL", use_column_width=True)
                user_instructions = st.text_area("Enter your instructions for what you need from this picture:",
                                               placeholder="e.g., Describe the scene, count the people, what objects are visible?",
                                               key="user_instructions_url")
            except:
                st.error("Failed to load image from URL")

    elif input_option == "Sample Images":
        sample_url = sample_images.get(selected_case, "https://ultralytics.com/images/bus.jpg")
        if st.button("Load Sample Image"):
            try:
                response = requests.get(sample_url, timeout=10)
                image = Image.open(io.BytesIO(response.content))
                st.image(image, caption="Sample Image", use_column_width=True)
                user_instructions = st.text_area("Enter your instructions for what you need from this picture:",
                                               placeholder="e.g., Describe the scene, count the people, what objects are visible?",
                                               key="user_instructions_sample")
            except:
                st.error("Failed to load sample image")

    elif input_option == "Local Test Images":
        local_images = get_local_test_images(selected_case)
        if local_images:
            selected_local_image = st.selectbox("Choose a local test image:", local_images)
            if selected_local_image and st.button("Load Selected Image"):
                try:
                    image = Image.open(selected_local_image)
                    st.image(image, caption=f"Local Test Image: {Path(selected_local_image).name}", use_column_width=True)
                    user_instructions = st.text_area("Enter your instructions for what you need from this picture:",
                                                   placeholder="e.g., Describe the scene, count the people, what objects are visible?",
                                                   key="user_instructions_local")
                except:
                    st.error("Failed to load local test image")
        else:
            st.info("No local test images found for this use case. Click '📥 Download Test Images' in the sidebar to get sample images.")

    elif input_option == "Video Upload":
        video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
        if video_file is not None:
            st.video(video_file)

# Process button and results
if st.button("🚀 Run YOLO Inference", type="primary"):
    if video_file is not None and selected_case == "🎥 Video Analytics":
        # Process video
        with st.spinner("Processing video... This may take a while."):
            task, dataset = task_mapping[selected_case]
            model = load_model(selected_model, task)

            # Save video temporarily
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(video_file.read())
                video_path = tmp.name

            try:
                # Run inference on video
                results = model(video_path, conf=confidence, save=True, project="results", name="video_output")

                # Show some results info
                st.success("Video processing complete!")
                st.write(f"Processed {len(results)} frames")
                if results and len(results) > 0:
                    with st.expander("📊 Detailed Results"):
                        st.json(results[0].tojson() if hasattr(results[0], 'tojson') else str(results[0]))

            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
            finally:
                os.unlink(video_path)

    elif image is not None:
        with st.spinner("Running inference..."):
            # Convert PIL to numpy array
            img_array = np.array(image)

            try:
                task, dataset = task_mapping[selected_case]
                model = load_model(selected_model, task)

                # Always run standard inference for detection data
                results = model(img_array, conf=confidence)

                # Special handling based on use case for metrics
                if selected_case == "👥 People Counting":
                    # Run detection and count people
                    people_count = 0
                    if results and len(results) > 0:
                        for r in results:
                            people_count += len([box for box, cls in zip(r.boxes, r.boxes.cls) if model.names[int(cls)] == 'person'])
                    st.metric("👥 People Count", people_count)

                elif selected_case == "🚶 Multi-Object Tracking":
                    # Track objects (note: tracking usually needs video, but can simulate on image)
                    results = model.track(img_array, conf=confidence, persist=True)
                    st.success("Tracking initialized (objects detected)")

                elif selected_case == "⚠️ Zone Intrusion Alerts":
                    # Define zones and check intrusion
                    zones = [(0.3, 0.3, 0.7, 0.7)]  # Example zone: central area
                    intrusions = []
                    if results and len(results) > 0:
                        for r in results:
                            for box in r.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                cx, cy = (x1 + x2) / 2 / img_array.shape[1], (y1 + y2) / 2 / img_array.shape[0]
                                for zone_x1, zone_y1, zone_x2, zone_y2 in zones:
                                    if zone_x1 <= cx <= zone_x2 and zone_y1 <= cy <= zone_y2:
                                        intrusions.append(model.names[int(box.cls)])
                    if intrusions:
                        st.warning(f"🚨 Intrusions detected: {', '.join(intrusions)}")
                    else:
                        st.success("✅ No intrusions detected")

                elif selected_case == "🛡️ Safety Compliance (PPE)":
                    # Check for safety equipment
                    ppe_items = ['helmet', 'helmet', 'vest', 'hardhat', 'safety vest']  # Assume detected classes
                    compliance = False
                    if results and len(results) > 0:
                        for r in results:
                            detected_classes = [model.names[int(cls)] for cls in r.boxes.cls]
                            if any(item in detected_classes for item in ppe_items):
                                compliance = True
                                break
                    if compliance:
                        st.success("🛡️ PPE Compliance: Safe")
                    else:
                        st.error("⚠️ PPE Compliance: Missing safety equipment")

                # Display results
                with col2:
                    st.header("📋 Results")
                    if results and len(results) > 0:
                        # Plot results
                        plotted_img = results[0].plot()
                        st.image(plotted_img, caption="Inference Results", use_column_width=True)

                        # Show detections in table
                        result_df = []
                        for r in results:
                            for box, cls, conf_val in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                                result_df.append({
                                    'Class': model.names[int(cls)],
                                    'Confidence': f"{conf_val:.2f}",
                                    'X1': f"{box[0]:.1f}",
                                    'Y1': f"{box[1]:.1f}",
                                    'X2': f"{box[2]:.1f}",
                                    'Y2': f"{box[3]:.1f}"
                                })

                        if result_df:
                            st.dataframe(pd.DataFrame(result_df))

                        # JSON export
                        with st.expander("📄 Raw JSON Results"):
                            st.json(results[0].tojson() if hasattr(results[0], 'tojson') else str(results[0]))

                        # Generate response based on user instructions
                        if user_instructions:
                            detections_summary = create_detections_summary(result_df, selected_case)
                            with st.spinner("Generating AI response based on your instructions..."):
                                system_prompt = f"You are an intelligent AI assistant analyzing a computer vision observation. The user provided these instructions: '{user_instructions}'.\n\nBased on the following YOLO object detection results from the task '{selected_case}':\n\n{detections_summary}\n\nProvide a clear, plain text answer following the user's instructions. Combine the instructions with the detection data to give a helpful response about what was observed in the image."
                                ai_response = query_ollama(system_prompt)
                                st.markdown(f"## 🤖 AI Response\n\n{ai_response}")
                        else:
                            st.info("Enter instructions for the image analysis to see AI response.")

            except Exception as e:
                st.error(f"Inference failed: {str(e)}")
                st.info("Make sure the model type matches the task. Some tasks require specific model variants.")

st.markdown("---")
st.markdown("**🔧 Tips:**")
st.markdown("- Use higher confidence thresholds for cleaner results")
st.markdown("- Try different model sizes (n=nano, x=xlarge) for speed vs accuracy trade-offs")
st.markdown("- For video analytics and tracking, use video input for full functionality")

# Dataset Information
display_dataset_info()

# Footer
st.markdown("---")
st.markdown("Built with Streamlit & Ultralytics YOLO 🚀 | Supports all YOLOv8 tasks and models")


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
