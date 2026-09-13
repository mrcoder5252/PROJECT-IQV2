"""
Script to generate sample academic documents (PDF, DOCX, PPTX)
with realistic sections, administrative front-matter, student PII, and technical gaps.
"""

from pathlib import Path
import fitz
from docx import Document
from pptx import Presentation
from pptx.util import Inches, Pt

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def create_sample_pdf():
    doc = fitz.open()

    # Page 1: Administrative Cover Page & Certificate
    page1 = doc.new_page()
    text_p1 = """SAVITRIBAI PHULE PUNE UNIVERSITY
DEPARTMENT OF COMPUTER ENGINEERING
WAGHOLI, PUNE 412207 2023-24

A PROJECT REPORT ON
SMART AGRICULTURE LEAF DISEASE DETECTION USING DEEP LEARNING

SUBMITTED BY:
VAIBHAV KADAM (Roll No: BE-COMP-42, PRN: 72018492K)
AMIT DESHMUKH (Roll No: BE-COMP-15, PRN: 72018431M)
Contact Email: vaibhav.kadam@college.edu | Phone: +91 9876543210

GUIDED BY:
PROF. S. K. SHARMA

THIS IS TO CERTIFY THAT the project report entitled "Smart Agriculture Leaf Disease Detection"
is a bonafide work carried out by the students in partial fulfillment of the degree.
"""
    page1.insert_text((50, 60), text_p1, fontsize=11)

    # Page 2: Abstract & Literature Survey
    page2 = doc.new_page()
    text_p2 = """ABSTRACT
Agriculture productivity is heavily impacted by plant crop diseases. Early detection of leaf anomalies
can prevent massive harvest losses. In this project, we develop an automated deep convolutional neural
network framework for diagnosing tomato and potato leaf blights.

LITERATURE SURVEY
Previous studies by LeCun et al. [1] established the efficacy of deep convolution backpropagation in visual recognition.
Tan and Le (2019) introduced EfficientNet architectures balancing parameter depth and FLOPs.
Recent work by Vaswani et al. [3] demonstrated self-attention mechanisms in sequence modeling.
However, existing agricultural imaging systems fail when deployed in uncontrolled outdoor lighting conditions.
"""
    page2.insert_text((50, 60), text_p2, fontsize=11)

    # Page 3: Methodology, Results, Future Scope, and References
    page3 = doc.new_page()
    text_p3 = """METHODOLOGY
Our proposed system architecture comprises three distinct stages:
1. Image Acquisition: Ingesting RGB leaf images captured at 1080p resolution.
2. Preprocessing & Augmentation: Normalization, random rotation, color jittering, and histogram equalization.
3. Classification Backbone: A custom ResNet-50 backbone trained with cross-entropy loss over 50 epochs.

RESULTS AND DISCUSSION
The model achieved 91.4% classification accuracy across 10 disease categories on the PlantVillage benchmark.
Training required 4 hours on an NVIDIA RTX 3080 GPU.

FUTURE SCOPE
1. High computational cost prevents real-time deployment on low-cost agricultural drones or Raspberry Pi microcontrollers.
2. Model performance drops significantly when background soil or weeds occlude the leaf margins.
3. Lack of federated edge updates means farmers cannot train local disease varieties without central server access.
4. Future work must investigate lightweight INT8 model quantization and multi-spectral edge imaging.

REFERENCES
[1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, 1998.
[2] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," ICML, 2019.
[3] A. Vaswani et al., "Attention is all you need," Advances in Neural Information Processing Systems, 2017.
"""
    page3.insert_text((50, 60), text_p3, fontsize=10)

    pdf_path = RAW_DIR / "smart_agriculture_blackbook.pdf"
    doc.save(str(pdf_path))
    doc.close()
    print(f"Created sample PDF: {pdf_path}")

def create_sample_docx():
    doc = Document()
    doc.add_heading("Healthcare IoT: Real-Time Patient Vital Monitoring and Fall Detection", level=0)

    # Front-matter / Author details
    p_meta = doc.add_paragraph()
    p_meta.add_run("Submitted by: Sneha Patil (Student ID: PRN9928120), Rohan Joshi\n")
    p_meta.add_run("Email: sneha.patil@healthtech.org | Mobile: +91 9123456780\n")
    p_meta.add_run("Guide: Dr. R. V. Kulkarni\n")

    doc.add_paragraph("ACKNOWLEDGEMENTS: We express our sincere gratitude to our department and mentors...")

    doc.add_heading("ABSTRACT", level=1)
    doc.add_paragraph(
        "Remote patient vital sign monitoring is critical for elderly individuals living independently. "
        "This project presents an embedded IoT wearable capable of streaming ECG, heart rate, and accelerometer telemetry."
    )

    doc.add_heading("LITERATURE REVIEW", level=1)
    doc.add_paragraph(
        "Chen et al. [2] explored wearable sensor network protocols for ambulatory arrhythmia detection. "
        "Gupta and Verma (2022) surveyed edge computing in telemedicine. Most existing devices suffer from rapid battery drain "
        "and lack encrypted on-device telemetry transmission."
    )

    doc.add_heading("METHODOLOGY", level=1)
    doc.add_paragraph(
        "The system incorporates an ESP32 microcontroller, AD8232 ECG sensor module, and MPU6050 accelerometer. "
        "Data is streamed via MQTT to a central dashboard with anomaly detection thresholding."
    )

    doc.add_heading("RESULTS AND DISCUSSION", level=1)
    doc.add_paragraph(
        "Prototype telemetry tests showed continuous ECG heart rate transmission with 98.2% beat detection accuracy "
        "over a 20-meter range in home environments with latency under 120ms."
    )

    doc.add_heading("FUTURE SCOPE", level=1)
    doc.add_paragraph(
        "Current hardware battery life is limited to 6 hours continuous streaming. "
        "The system cannot operate in remote rural areas without reliable Wi-Fi or cellular connectivity. "
        "Further research is required to implement energy-harvesting Bluetooth Low Energy (BLE) protocols and on-chip TinyML inference."
    )

    doc.add_heading("REFERENCES", level=1)
    doc.add_paragraph("[1] J. Chen, K. Atallah, and W. Wang, 'Wearable ECG sensor telemetry protocols,' IEEE Trans. Biomed. Eng., 2021.")
    doc.add_paragraph("[2] A. Gupta and S. Verma, 'Edge computing architectures for tele-health monitoring,' ACM Computing Surveys, 2022.")

    docx_path = RAW_DIR / "healthcare_iot_research_paper.docx"
    doc.save(str(docx_path))
    print(f"Created sample DOCX: {docx_path}")

def create_sample_pptx():
    prs = Presentation()

    # Slide 1: Title & Presentation details
    slide1 = prs.slides.add_slide(prs.slide_layouts[0])
    slide1.shapes.title.text = "Decentralized Blockchain Land Registry"
    slide1.placeholders[1].text = "Review 4 Presentation\nTeam Members: Aniket Shinde, Pooja Nair\nGuide: Prof. Patil"

    # Slide 2: Problem Statement & Objectives
    slide2 = prs.slides.add_slide(prs.slide_layouts[1])
    slide2.shapes.title.text = "PROBLEM STATEMENT & OBJECTIVES"
    slide2.placeholders[1].text = "Traditional municipal land record registries suffer from fraud, tampering, and bureaucratic delays. Objective is to create a tamper-evident Ethereum smart contract registry."

    # Slide 3: Proposed Architecture
    slide3 = prs.slides.add_slide(prs.slide_layouts[1])
    slide3.shapes.title.text = "SYSTEM ARCHITECTURE"
    slide3.placeholders[1].text = "1. Frontend React Web3 portal.\n2. Solidity Smart Contract for land deed tokenization (ERC-721).\n3. InterPlanetary File System (IPFS) for encrypted deed document storage."

    # Slide 4: Results and Prototype Evaluation
    slide4 = prs.slides.add_slide(prs.slide_layouts[1])
    slide4.shapes.title.text = "RESULTS & PERFORMANCE EVALUATION"
    slide4.placeholders[1].text = "Successfully deployed testnet contract on Sepolia. Deed minting confirmed in 14 seconds average block time with verified IPFS hash provenance."

    # Slide 5: Conclusion & Future Scope
    slide5 = prs.slides.add_slide(prs.slide_layouts[1])
    slide5.shapes.title.text = "FUTURE SCOPE & LIMITATIONS"
    slide5.placeholders[1].text = "Limitations: High Ethereum mainnet gas transaction costs and lack of cross-chain interoperability with government legacy SQL databases. Future scope should incorporate Layer-2 zero-knowledge rollups (zk-SNARKs) and automated Oracle price feeds."

    pptx_path = RAW_DIR / "blockchain_review4_presentation.pptx"
    prs.save(str(pptx_path))
    print(f"Created sample PPTX: {pptx_path}")

if __name__ == "__main__":
    create_sample_pdf()
    create_sample_docx()
    create_sample_pptx()
    print("All sample files generated successfully in data/raw/")
