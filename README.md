# Distributed Machine Learning Systems Course Projects

<div align="center">

![University of Tehran](https://img.shields.io/badge/University%20of-Tehran-blue)
![Course](https://img.shields.io/badge/Course-DMLS-success)
![Instructor](https://img.shields.io/badge/Instructor-Dr.%20Mohammad%20Javad%20Dousti-orange)
![Python](https://img.shields.io/badge/Python-3.x-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Apache Spark](https://img.shields.io/badge/Apache-Spark-brightgreen)
![CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900)
![MPI](https://img.shields.io/badge/MPI-Distributed%20Computing-blueviolet)

</div>

---

## Overview

This repository contains my solutions, implementations, experiments, and reports for the **Distributed Machine Learning Systems (DMLS)** course offered by the **Department of Electrical and Computer Engineering, University of Tehran**, under the supervision of **Dr. Mohammad Javad Dousti**.

The course focuses on the principles and practical aspects of building scalable machine learning systems. Throughout the assignments, various distributed computing paradigms and modern AI infrastructures were explored, ranging from federated learning and MPI-based parallel computing to GPU acceleration, Apache Spark analytics, distributed neural network training, and performance profiling.

Each assignment includes source code, implementation details, experimental results, visualizations, and comprehensive technical reports.

---

## Table of Contents

* [Repository Structure](#repository-structure)
* [Computer Assignment 1 - Federated Learning](#computer-assignment-1---federated-learning)
* [Computer Assignment 2 - CUDA and Distributed Deep Learning](#computer-assignment-2---cuda-and-distributed-deep-learning)
* [Computer Assignment 3 - Apache Spark and Large-Scale Data Analytics](#computer-assignment-3---apache-spark-and-large-scale-data-analytics)
* [Computer Assignment 4 - Distributed Training and Profiling](#computer-assignment-4---distributed-training-and-profiling)
* [Technologies Used](#technologies-used)
* [Course Information](#course-information)

---

## Repository Structure

```text
.
├── CA1/
│   ├── Source Codes
│   ├── Experimental Results
│   ├── Plots
│   └── Report
│
├── CA2/
│   ├── CUDA Kernels
│   ├── CNN Training
│   ├── Distributed Execution
│   └── Report
│
├── CA3/
│   ├── Video Game Analytics
│   ├── Word2Vec
│   ├── HDFS
│   └── Report
│
├── CA4/
│   ├── Distributed Data Parallel
│   ├── Accelerate
│   ├── Profiling
│   ├── SLURM Scripts
│   └── Report
│
└── README.md
```

---

# Computer Assignment 1 - Federated Learning

### Topics

* Federated Averaging (FedAvg)
* Decentralized Training
* Client-Server Learning Paradigm
* Communication-Efficient Learning
* Model Aggregation Strategies

### Description

The first assignment introduces the fundamentals of **Federated Learning**, a distributed machine learning paradigm in which multiple clients collaboratively train a shared model without exchanging their private data. Instead of transferring raw datasets to a centralized server, each client performs local training and periodically sends model updates to an aggregation server.

In this assignment, several federated learning configurations were implemented and analyzed. Different numbers of clients, communication rounds, and aggregation strategies were investigated to evaluate their impact on model convergence, training stability, communication overhead, and final predictive performance. Experimental results provide valuable insights into the trade-offs between privacy preservation, communication cost, and learning efficiency in large-scale distributed environments.

---

# Computer Assignment 2 - CUDA and Distributed Deep Learning

### Topics

* CUDA Programming
* GPU Acceleration
* Parallel Kernel Design
* CNN Training
* Distributed Deep Learning

### Description

The second assignment focuses on leveraging GPU architectures to accelerate machine learning workloads. Several CUDA-based implementations were developed to better understand how modern deep learning frameworks utilize massively parallel hardware resources.

The assignment begins with low-level CUDA programming concepts, including thread organization, memory hierarchy, and kernel optimization. Afterwards, these concepts are applied to deep learning workloads, where convolutional neural networks are trained and evaluated under different execution settings. The project investigates how GPU acceleration significantly reduces computational latency while improving throughput for large-scale machine learning tasks. Performance comparisons between CPU and GPU executions demonstrate the benefits of parallel processing in practical AI systems.

---

# Computer Assignment 3 - Apache Spark and Large-Scale Data Analytics

### Topics

* Apache Spark
* RDD Programming
* Spark MLlib
* HDFS
* Large-Scale Data Processing

### Description

The third assignment explores large-scale data processing using the Apache Spark ecosystem. The objective is to understand how modern distributed data analytics frameworks handle datasets that exceed the capabilities of traditional single-machine solutions.

Several practical case studies were implemented. These include exploratory analysis of large video game sales datasets, distributed training of Word2Vec embeddings for natural language processing tasks, and the development of machine learning pipelines on datasets stored in HDFS. Through these experiments, concepts such as resilient distributed datasets (RDDs), distributed transformations, fault tolerance, feature engineering, and scalable machine learning were investigated. The assignment demonstrates how Spark enables efficient analytics and learning over large volumes of structured and unstructured data.

---

# Computer Assignment 4 - Distributed Training and Profiling

### Topics

* PyTorch Distributed Data Parallel (DDP)
* Hugging Face Accelerate
* Multi-GPU Training
* Profiling and Performance Analysis
* SLURM-based Execution

### Description

The final assignment investigates modern frameworks for distributed deep learning and performance optimization. The first part focuses on implementing distributed training using PyTorch Distributed Data Parallel (DDP), where multiple GPUs collaboratively train neural networks while synchronizing gradients efficiently. The second part examines Hugging Face Accelerate as a higher-level framework that simplifies distributed execution while maintaining scalability and performance.

A dedicated profiling section was also developed to analyze computational bottlenecks, GPU utilization, memory consumption, kernel execution behavior, and communication overhead. Using profiling tools and performance traces, the training workflow was carefully evaluated to identify inefficiencies and understand the runtime characteristics of distributed machine learning systems. These experiments provide practical experience in optimizing large-scale AI workloads deployed on modern computing infrastructures.

---

## Technologies Used

* Python
* NumPy
* PyTorch
* CUDA
* MPI
* Apache Spark
* Spark MLlib
* HDFS
* Hugging Face Accelerate
* Distributed Data Parallel (DDP)
* SLURM
* Jupyter Notebook

---

## Course Information

**Course:** Distributed Machine Learning Systems (DMLS)

**Instructor:** Dr. Mohammad Javad Dousti

**Department:** Electrical and Computer Engineering

**University:** University of Tehran

This repository is intended for educational, research, and learning purposes and showcases practical implementations of distributed machine learning algorithms and scalable AI infrastructures.

## ⭐️ Support

If you find this repository useful, consider giving it a ⭐️
