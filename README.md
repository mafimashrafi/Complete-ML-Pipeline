# Complete ML Pipeline

This project is an experiment in implementing an end-to-end (e2e) Machine Learning pipeline and tracking data versions using **DVC (Data Version Control)** with **AWS S3** as remote storage.

## 📁 Project Structure

```text
Complete-ML-Pipeline/
│
├── .dvc/                  # DVC configuration and metadata
├── dvclive/               # Metrics, parameters, and evaluation logs
├── experiments/           # Jupyter notebooks and experimental scripts
├── src/                   # Source code for pipeline stages (ingestion, training, etc.)
│
├── .dvcignore             # Files to ignore in DVC
├── .gitignore             # Git ignore rules
├── dvc.lock               # DVC lock file locking data and stage outputs
├── dvc.yaml               # DVC pipeline definition (automated stages)
├── params.yaml            # Hyperparameters and configuration parameters
├── ProjectFlow.txt        # Project roadmap and planning notes
└── README.md              # Project documentation

##
