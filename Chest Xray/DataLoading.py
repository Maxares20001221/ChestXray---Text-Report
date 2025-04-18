# Data Extraction & Loading Process for Google Drive Environment

from google.colab import drive
import tarfile
import os
import os
import xml.etree.ElementTree as ET
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

drive.mount('.../drive')
# Target path
image_dir = ".../images"
report_dir = ".../reports"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

# Extract function
def extract_tgz(tgz_path, dest_path):
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(dest_path)
        print(f"Extracted to {dest_path}")

# Decompress two tgz files
extract_tgz(".../NLMCXR_png.tgz", image_dir)
extract_tgz(".../NLMCXR_reports.tgz", report_dir)


# Visualize the first image
# Image path
img_path = ".../CXR799_IM-2333-1001.png"
# Load and display the image
img = Image.open(img_path)
plt.imshow(img, cmap='gray')
plt.title("CXR799_IM-2333-1001.png")
plt.axis('off')
plt.show()


# Create data pairs
# Path of image and report
image_root = ".../images"
report_root = ".../ecgen-radiology"

data_pairs = []

def extract_report_text(xml_path):
    """
    Extract the findings and impressions from an XML report.

    Args:
        xml_path (str): Path to the XML report.

    Returns:
        str: Combined findings and impressions.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        findings, impression = "", ""

        for abstract in root.iter("AbstractText"):
            label = abstract.attrib.get("Label", "").lower()
            if label == "findings":
                findings = abstract.text.strip() if abstract.text else ""
            elif label == "impression":
                impression = abstract.text.strip() if abstract.text else ""

        report = findings + " " + impression
        return report.strip()
    except Exception as e:
        print(f"[Parse error] {xml_path}: {e}")
        return ""

# Load all the paths of the files
image_paths = glob(os.path.join(image_root, "*.png"))

# Iterate all xml reports
for xml_file in glob(os.path.join(report_root, "*.xml")):
    report_text = extract_report_text(xml_file)
    if report_text:
        file_id = os.path.splitext(os.path.basename(xml_file))[0]
        image_prefix = f"CXR{file_id}_"
        matched_images = [img_path for img_path in image_paths if image_prefix in os.path.basename(img_path)]
        if matched_images:
            data_pairs.append((matched_images[0], report_text))

# Show first 5 samples in data_pairs
print("==== Preview of First 5 (image, report) pairs ====\n")
for idx, (img_path, report_text) in enumerate(data_pairs[:5]):
    print(f"[Sample {idx + 1}]")
    print("Image Path:", img_path)
    print("Report:\n", report_text)
    print("-" * 60)

# Display the image with reports
print(f"Collected {len(data_pairs)} valid (image, report) pairs.")

if data_pairs:
    img_path, report = data_pairs[0]
    print("\nSample Image Path:", img_path)
    print("Report:\n", report)

    image = Image.open(img_path)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.title("Chest X-Ray")
    plt.show()
