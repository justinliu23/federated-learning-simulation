# federated-learning-simulation

This repository contains a Federated Learning (FL) simulation developed in Python, designed to mimic federated training on a single machine using a CNN architecture. The simulation involves creating multiple devices that each train independently on local data, sending only model updates to a central server. This server then averages these updates to produce a global model, enabling privacy-preserving collaborative learning without centralized data collection.

## Project Overview

Federated Learning (FL) allows multiple clients (devices) to train a machine learning model collaboratively without sharing their private data. This is accomplished through:
1. **Local Training**: Each device independently trains a model using its local dataset.
2. **Model Aggregation**: Devices send only the model parameters (weights) to a central server.
3. **Federated Averaging**: The server aggregates these model updates to produce a globally shared model, which is then redistributed for further training rounds.

This simulation implements a FL environment that supports both IID (Independent and Identically Distributed) and non-IID data distributions across devices, using the following features.

## Key Features

- **Device Simulation**: Each device simulates a unique client, containing:
  - `ConvNet` architecture: A CNN model defined for image classification.
  - DataLoader: Loads local data for each device, partitioned according to IID or non-IID settings.
  - Optimizer and Scheduler: Configured to handle local training and dynamic learning rate adjustments.

- **Dataset Partitioning**:
  - `DatasetSplit`: A class that supports data partitioning into subsets for each simulated device.
  - `iid_sampler`: A function that enables IID data distribution across devices, ensuring each device’s data reflects the global distribution.
  - **Non-IID Handling**: The code can be modified for non-IID distribution by customizing `data_idxs` for each device, which allows simulating more realistic, unbalanced data scenarios across devices.

- **Federated Averaging**:
  - The `average_weights` function aggregates model weights from each device after a training round. This function implements the FedAvg algorithm, which averages device updates to produce the global model.

- **Training Simulation**:
  - The code cycles between local training on devices and centralized weight aggregation. 
  - Each device’s test accuracy is tracked in `test_acc_tracker` across rounds, providing insights into model performance and convergence.

## IID vs Non-IID Data Handling

In federated learning, **IID (Independent and Identically Distributed)** and **non-IID** setups impact model performance and stability. This simulation explores both setups:
  
- **IID Data Distribution**: 
  - Using `iid_sampler`, data is uniformly distributed across devices. Each device receives a subset that represents the overall dataset distribution, ensuring consistent updates from all devices. This setup is ideal for stable convergence and balanced performance across all data types.

- **Non-IID Data Distribution**: 
  - In real-world federated learning, devices often hold data with unique distributions, reflecting individual user patterns. This non-IID setting can be simulated by assigning different, potentially imbalanced data subsets to each device. For instance, one group may have data dominated by certain classes while other groups see different distributions.

## Core Functions

- **`create_device`**: Initializes each device with a copy of the `ConvNet` model, data loader, optimizer, and scheduler.
- **`average_weights`**: Implements federated averaging by calculating the mean of device weights.
- **`get_devices_for_round`**: Randomly selects devices for each training round, supporting flexible participation rates.
- **`iid_sampler`**: Partitions the dataset for IID settings, ensuring a balanced data distribution across devices.

### Requirements
- Python 3.6+
- Required libraries: `torch`, `torchvision`

### Steps to Run
1. Clone the repository.
2. Configure parameters as needed (e.g., `data_pct`, `num_devices`).
3. Run the code to initiate the federated learning simulation.

## Insights and Results

The results underscore the importance of considering data distribution in federated learning:

- **Unfair Device Participation**: Overrepresented groups achieve higher accuracy, leading to bias in the global model.
- **Fluctuating Accuracy in Fair Participation**: In fair participation scenarios, non-IID data causes accuracy to vary considerably across rounds, as updates from different groups intermittently dominate.
