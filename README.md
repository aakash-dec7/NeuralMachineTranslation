# Neural Machine Translation

This repository contains a **Transformer-based Neural Machine Translation** (NMT) model implemented in `PyTorch`. The model follows the original Transformer architecture introduced by Vaswani et al. in *Attention is All You Need*. This implementation includes essential components such as multi-head attention, positional encoding, and feed-forward networks.

## Features

**PyTorch Implementation**: Fully built using `PyTorch` for efficient deep learning workflows.
**Data Version Control (DVC)**: Manages training and evaluation pipelines effectively.
**Marian Tokenizer**: Uses `Helsinki-NLP/opus-mt-en-fr` for robust and efficient text processing.
**Experiment Tracking & Model Management**: Integrated with `MLflow` and `DagsHub` for seamless tracking.
**Containerized Deployment**: Docker images stored in `Amazon Elastic Container Registry (ECR)`.
**Scalable Deployment**: Model deployed on `Amazon Elastic Kubernetes Service (EKS)` for production readiness.
**Automated CI/CD**: End-to-end deployment automation using AWS and GitHub Actions.

## Prerequisites

Ensure the following dependencies and services are installed and configured:

- Python 3.10
- AWS Account
- AWS CLI
- Docker Desktop (for local testing)
- DagsHub Account (for experiment tracking)
- Git & GitHub (for version control)

## Dataset

**Source**: [Language Translation (English-French)](https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench)

**Description**:

The dataset consists of two columns:

- **English Sentences**
- **French Translations**

## Model Architecture

The **NeuralMachineTranslation** model consists of the following components:

### 1. Positional Encoding

- Since Transformers lack recurrence mechanisms, **positional encodings** are added to retain order information.
- Uses `sine` and `cosine` functions for unique position encodings.

### 2. Multi-Head Attention

- Implements `self-attention` and `cross-attention` mechanisms.
- Splits the input into multiple **attention heads** for better feature representation.
- Uses **scaled dot-product attention** to compute `attention scores`.

### 3. Feed-Forward Network (FFN)

- Applies two **linear transformations** with a `ReLU` activation.
- Enhances model expressiveness.

### 4. Encoder Layer

- Contains a **multi-head self-attention** mechanism.
- Followed by a **position-wise feed-forward network**.
- Uses `layer normalization` after each sub-layer.

### 5. Decoder Layer

- Similar to the encoder but includes an additional **multi-head cross-attention mechanism**.
- Attends to the **encoder's output** while processing the **target sequence**.
- Uses masked `self-attention` to prevent future token leakage.

### 6. Transformer Model

- Stacks multiple **encoder** and **decoder** layers.
- Uses `embedding` layers for input representation.
- Outputs token probabilities for translation.

#### Model Summary

```text
Model(
  (embedding): Embedding(59515, 512)
  (positional_encoding): PositionalEncoding()
  (encoder_layers): ModuleList(
    (0-1): 2 x EncoderLayer(
      (attention): MultiheadAttention(
        (w_q): Linear(in_features=512, out_features=512, bias=False)
        (w_k): Linear(in_features=512, out_features=512, bias=False)
        (w_v): Linear(in_features=512, out_features=512, bias=False)
        (w_o): Linear(in_features=512, out_features=512, bias=False)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (ffnn): FeedForward(
        (net): Sequential(
          (0): Linear(in_features=512, out_features=1024, bias=True)
          (1): ReLU()
          (2): Dropout(p=0.2, inplace=False)
          (3): Linear(in_features=1024, out_features=512, bias=True)
        )
      )
      (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.2, inplace=False)
    )
  )
  (decoder_layers): ModuleList(
    (0-1): 2 x DecoderLayer(
      (self_attention): MultiheadAttention(
        (w_q): Linear(in_features=512, out_features=512, bias=False)
        (w_k): Linear(in_features=512, out_features=512, bias=False)
        (w_v): Linear(in_features=512, out_features=512, bias=False)
        (w_o): Linear(in_features=512, out_features=512, bias=False)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (cross_attention): MultiheadAttention(
        (w_q): Linear(in_features=512, out_features=512, bias=False)
        (w_k): Linear(in_features=512, out_features=512, bias=False)
        (w_v): Linear(in_features=512, out_features=512, bias=False)
        (w_o): Linear(in_features=512, out_features=512, bias=False)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (ffnn): FeedForward(
        (net): Sequential(
          (0): Linear(in_features=512, out_features=1024, bias=True)
          (1): ReLU()
          (2): Dropout(p=0.2, inplace=False)
          (3): Linear(in_features=1024, out_features=512, bias=True)
        )
      )
      (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.2, inplace=False)
    )
  )
  (fc_out): Linear(in_features=512, out_features=59515, bias=True)
  (dropout): Dropout(p=0.2, inplace=False)
)
```

## Installation

Clone the repository:

```sh
git clone https://github.com/aakash-dec7/NeuralMachineTranslation.git
cd NeuralMachineTranslation
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Initialize DVC Pipeline

```sh
dvc init
```

## DVC Pipeline Stages

1. **Data Ingestion** - Fetches and stores the raw dataset.
2. **Data Validation** - Ensures data quality and integrity before processing.
3. **Data Preprocessing** - Cleans, tokenizes, and prepares the dataset.
4. **Data Transformation** - Converts processed data for model training.
5. **Model Definition** - Defines the Transformer model architecture.
6. **Model Training** - Trains the model using the transformed dataset.
7. **Model Evaluation** - Assesses the trained model’s performance.

### Run Model Training and Evaluation

```sh
dvc repro
```

The trained model is saved in `artifacts/model/model.pth`.

## Deployment

### Create an ECR Repository

Create an Amazon ECR repository with the specified name in `setup.py`:

```python
setup(
    name="nmt",
    version="1.0.0",
    author="Aakash Singh",
    author_email="aakash.dec7@gmail.com",
    packages=find_packages(),
)
```

### Create an EKS Cluster

Use the following command to create an Amazon EKS cluster:

```sh
eksctl create cluster --name <cluster-name> \
    --region <region> \
    --nodegroup-name <nodegroup-name> \
    --nodes <number-of-nodes> \
    --nodes-min <nodes-min> \
    --nodes-max <nodes-max> \
    --node-type <node-type> \
    --managed
```

### Push Code to GitHub

Before pushing the code, add necessary GitHub Actions secrets under **Settings > Secrets and Variables > Actions**:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `ECR_REGISTRY_URI`

Push the code to GitHub:

```sh
git add .
git commit -m "Initial commit"
git push origin main
```

### CI/CD Automation

GitHub Actions automates CI/CD processes, ensuring the model is built, tested, and deployed on EKS.

## Accessing the Deployed Application

Once deployment is successful:

1. Navigate to **EC2 Instances** in the **AWS Console**.
2. Update inbound rules in **Security Groups** to allow traffic.

Retrieve the external IP of the deployed service:

```sh
kubectl get svc
```

Copy the `EXTERNAL-IP` and append `:5000` to access the application:

```text
http://<EXTERNAL-IP>:5000
```

The NeuralMachineTranslation application is now deployed and accessible online.

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
