<div align="center">
MedCore AI Labs
Distributed Medical AI Infrastructure

High-performance DICOM ingestion, PHI sanitization, spatial normalization, volumetric tensor processing, and distributed GPU execution for clinical research and medical AI workloads.

<br>










<br>

MRI · CT · PET · PyTorch · CUDA · HPC · Distributed GPU

</div>
Overview

MedCore AI Labs develops infrastructure for transforming heterogeneous clinical imaging data into standardized, machine-learning-ready volumetric tensors.

The medcore-dicom-pipeline is designed to bridge legacy radiology systems and modern AI inference infrastructure by providing a structured processing pipeline for:

Multi-vendor DICOM ingestion
Automated PHI metadata sanitization
Volumetric reconstruction
Spatial normalization
FP32 / FP16 tensor conversion
CUDA-accelerated processing
Multi-node GPU execution
Persistent PyTorch tensor artifacts

The system is intended for clinical research, medical AI engineering, dataset preprocessing, and HPC workloads.

Architecture

The pipeline follows a staged processing architecture:

Legacy DICOM SourcesMRI · CT · PET
DICOM Ingestion
PHI SanitizationMetadata Scrubbing
Volumetric Reconstruction
Spatial Normalization1.0 × 1.0 × 1.0 mm
Tensor ConversionFP32 / FP16
Execution Backend
NVIDIA CUDAA100 · A10 · T4
Multi-threaded CPU
Persistent Tensor Artifacts.pt
AI Training / InferenceCNN · ViT · HPC
Processing lifecycle
Discover clinical DICOM files and series.
Parse metadata and pixel data using pydicom.
Sanitize configured sensitive metadata fields.
Reconstruct volumetric image data from individual slices.
Normalize spatial resolution across heterogeneous scanners.
Convert the resulting volume into PyTorch tensors.
Accelerate execution through CUDA when available.
Persist optimized tensor artifacts for downstream workloads.
Core Capabilities
DICOM Ingestion
Recursive directory scanning
Standard .dcm files
Extensionless clinical series
Multi-vendor datasets
Volumetric slice reconstruction
Metadata-aware processing
PHI Sanitization

The sanitization layer targets sensitive DICOM attributes including:

PatientName
PatientID
PatientBirthDate
InstitutionName
PhysiciansOfRecord


The processing model is designed to ensure metadata sanitization occurs before downstream tensor persistence.

Spatial Normalization

Clinical datasets may originate from scanners with different voxel spacing and acquisition characteristics.

The pipeline supports standardized isotropic spatial calibration:

Target spacing:
1.0 × 1.0 × 1.0 mm


This provides a consistent spatial representation for downstream volumetric neural networks.

Tensor Conversion

The pipeline converts reconstructed volumes into PyTorch-compatible tensors supporting:

FP32
FP16
Configurable target dimensions
Volumetric N × C × D × H × W layouts
GPU execution
CPU fallback
Tensor Representation

The standard volumetric representation is:

(N, C, D, H, W)

N = Batch
C = Channels
D = Depth
H = Height
W = Width


Example:

target_shape = (1, 1, 64, 256, 256)


Resulting tensor:

Batch:       1
Channels:    1
Depth:      64
Height:    256
Width:     256


This representation is suitable for 3D convolutional architectures and volumetric vision-transformer pipelines.

Security & Data Governance

MedCore AI Labs uses a zero-trust processing model in which sensitive metadata is handled explicitly before downstream artifact generation.

The security layer is designed around:

PHI minimization
Metadata sanitization
Controlled data processing
Cryptographic audit logging
Encryption-aware infrastructure
Access-controlled storage
Dataset provenance
Retention policies
Audit Trail

The architecture supports cryptographic audit mechanisms based on:

SHA-256
AES-256


These mechanisms are intended to provide verifiable processing records and infrastructure-level auditability.

Important: Technical PHI sanitization does not, by itself, establish legal or regulatory compliance. HIPAA compliance depends on the complete operational, organizational, administrative, physical, and technical controls of the deployment environment.

Compute Infrastructure

The compute layer supports heterogeneous execution environments.

Layer	Supported Configuration
GPU Runtime	NVIDIA CUDA 12.x
GPU Architectures	A100 / A10 / T4
CPU Runtime	Multi-threaded fallback
Tensor Framework	PyTorch
Numerical Layer	NumPy
DICOM Engine	pydicom
Operating System	Ubuntu 22.04 LTS
Python	3.9+
PyTorch	2.0+

For current PyTorch installation commands, use the official PyTorch installer because the recommended package varies by operating system and CUDA configuration.

Distributed Execution

The architecture is designed for decentralized cloud and HPC environments.

DICOM Data
Ingestion Node
Processing Queue
GPU Node 01NVIDIA A100
GPU Node 02NVIDIA A10
GPU Node NNVIDIA T4
Persistent Storage
Tensor Artifacts
AI Training / Inference

The distributed execution model is intended to reduce processing latency between:

ingestion → transformation → acceleration → persistence

and to support horizontally scalable compute environments.

Repository Structure
medcore-dicom-pipeline/
├── main.py
├── README.md
├── requirements.txt
├── sample_data/
│   └── ct_mri_scans/
├── output/
│   └── persistent_tensors/
└── tests/

Installation
Requirements
Python 3.9+
PyTorch 2.0+
NumPy
pydicom
CUDA 12.x for NVIDIA GPU acceleration
Ubuntu 22.04 LTS recommended for GPU cluster deployments
Install dependencies
pip install pydicom numpy torch torchvision


For CUDA-enabled installations, select the PyTorch build appropriate for the target NVIDIA/CUDA environment using the official PyTorch installation selector.

Quick Start
Clone
git clone https://github.com/<your-org>/medcore-dicom-pipeline.git
cd medcore-dicom-pipeline

Install
pip install pydicom numpy torch torchvision

Prepare data
sample_data/
└── ct_mri_scans/
    ├── series_001/
    ├── series_002/
    └── ...

Execute
python main.py

Programmatic Usage
from main import DICOMProcessor

processor = DICOMProcessor(
    input_dir="./sample_data/ct_mri_scans",
    output_dir="./output/persistent_tensors",
    target_spacing=(1.0, 1.0, 1.0)
)

# Discover DICOM files
discovered_files = processor.scan_input_directory()

# Sanitize sensitive metadata
processor.sanitize_metadata(
    remove_patient_id=True
)

# Convert volumetric data
tensor_artifact = processor.convert_to_tensor(
    dtype="float32",
    target_shape=(1, 1, 64, 256, 256)
)

# Persist tensor artifact
saved_path = processor.export_tensor_to_cluster(
    tensor_artifact,
    filename="volumetric_mri_tensor_prod.pt"
)

print(f"Discovered files: {len(discovered_files)}")
print(f"Tensor shape: {tuple(tensor_artifact.shape)}")
print(f"Saved artifact: {saved_path}")

API Surface

The primary processing engine is exposed through DICOMProcessor:

class DICOMProcessor:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        target_spacing: Tuple[float, float, float] = (
            1.0, 1.0, 1.0
        )
    ):
        ...

    def scan_input_directory(self) -> List[str]:
        ...

    def sanitize_metadata(
        self,
        remove_patient_id: bool = True
    ) -> bool:
        ...

    def convert_to_tensor(
        self,
        dtype: str = "float32",
        target_shape: Tuple[int, int, int, int, int] = (
            1, 1, 64, 256, 256
        )
    ) -> torch.Tensor:
        ...

    def export_tensor_to_cluster(
        self,
        tensor: torch.Tensor,
        filename: str
    ) -> str:
        ...

Output Artifacts

Processed volumes can be persisted as PyTorch .pt artifacts:

output/
└── persistent_tensors/
    ├── volumetric_mri_tensor_prod.pt
    ├── volumetric_ct_tensor_prod.pt
    └── volumetric_pet_tensor_prod.pt


These artifacts can be consumed by downstream:

3D CNN pipelines
Vision Transformers
Medical image segmentation systems
Classification models
Training pipelines
Distributed inference systems
HPC batch workloads
Performance & Verification
Operational Phase	Target	Status
DICOM Ingestion	Siemens / GE / Philips compatibility	Verified
PHI Sanitization	Sensitive identifier removal	Enforced
Audit Logging	SHA-256 cryptographic logging	Enforced
Spatial Calibration	1.0 × 1.0 × 1.0 mm target	Enabled
Tensor Conversion	FP32 / FP16	Supported
GPU Execution	NVIDIA CUDA	Active
GPU Topology	A100 / A10 / T4	Supported
CPU Fallback	Multi-threaded processing	Supported

Benchmarking note: Performance targets should be independently validated against the exact dataset size, modality, scanner protocol, storage subsystem, CPU topology, GPU model, CUDA version, and software configuration used in production.

Supported Medical Imaging Workloads

The pipeline is designed for:

MRI volumetric preprocessing
CT reconstruction and preprocessing
PET volumetric processing
3D image classification
3D segmentation
Medical image detection
Multi-modal imaging research
Vision Transformer preprocessing
Large-scale AI dataset preparation
Distributed inference
Clinical Data Governance

Production deployments should establish appropriate controls for:

Identity and access management
Encryption at rest
Encryption in transit
Key management
Dataset provenance
Data retention
Audit-log retention
Infrastructure monitoring
Network segmentation
Institutional security requirements
Regulatory review
Clinical validation

The software infrastructure should be integrated into the organization's broader clinical data governance and security program rather than treated as a standalone compliance mechanism.

Development

Run the processing node locally:

python main.py


Run tests when available:

pytest


For GPU environments, verify CUDA visibility before executing large workloads:

import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

Roadmap

Potential future infrastructure capabilities include:

 DICOMweb ingestion
 Object-storage backends
 Kubernetes-based GPU scheduling
 Distributed job queues
 Multi-GPU tensor sharding
 Containerized deployment
 Automated dataset validation
 Enhanced DICOM conformance checks
 Model-serving integration
 Observability and metrics
 Institutional identity integration
Enterprise Integration

MedCore AI Labs infrastructure is intended to integrate with:

Clinical Systems
       │
       ▼
DICOM / DICOMweb
       │
       ▼
MedCore Processing Layer
       │
       ├── PHI Sanitization
       ├── Spatial Normalization
       ├── Tensor Conversion
       └── GPU Acceleration
       │
       ▼
HPC / Cloud Compute
       │
       ▼
Persistent Tensor Storage
       │
       ▼
Medical AI Workloads


For institutional infrastructure integration, technical documentation, cluster deployment guidance, node architecture configuration, or grant evaluation audits:

MedCore AI Labs — Core Infrastructure Engineering Team

infrastructure@medcore-ai.xyz

License

MedCore AI Labs Core Infrastructure Engineering Team. All rights reserved.

License information should be added here before public distribution of the repository.

Disclaimer

This project is intended for clinical research, engineering, infrastructure, and medical AI development workflows.

It is not, by itself:

a certified medical device;
a clinical diagnostic system;
a substitute for institutional validation;
a substitute for regulatory review;
a substitute for professional medical judgment.

All production deployments should undergo appropriate security, privacy, clinical, and regulatory assessment.

<div align="center">
MedCore AI Labs

Distributed infrastructure for next-generation medical AI.

DICOM → Sanitization → Normalization → Tensor → GPU → AI

</div> :::{"fallbackMarkdown":"","reference":{"matched_text":" ","prefix":null,"start_idx":15657,"end_idx":15657,"safe_urls":[],"refs":[],"alt":"","prompt_text":null,"type":"sources_footnote","sources":[{"title":"Quickstart for writing on GitHub - GitHub Docs","url":"https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/quickstart-for-writing-on-github?utm_source=chatgpt.com","attribution":"GitHub Docs"},{"title":"Get Started","url":"https://pytorch.org/get-started/locally/?utm_source=chatgpt.com","attribution":"PyTorch"}],"has_images":false},"showLoginRequiredCard":false}
