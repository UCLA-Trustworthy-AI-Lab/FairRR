# FairRR
Codebase and Experiments for FairRR: Pre-Processing for Group Fairness through Randomized Response

## Table of Contents

- [Codebase Overview](#codebase-overview)
- [Getting Started](#getting-started)
- [Data](#data)


## Codebase Overview
The implementation of FairRR and other benchmarking methods can be found in algorithm.py. To replicate the experiments in FairRR: Pre-Processing for Group Fairness through Randomized Response run main.py which will train, test, and save the results for each method across datasets. After the raw results are generated, run analyze.py to process the results. Myfunctions.py and dataloader.py include helper functions.

## Getting Started

To run the code and experiments in this repository, you'll need to set up your environment and install the necessary dependencies. Follow the steps below:

1. **Clone the Repository:**
  
   ```bash
   git clone https://github.com/yourusername/your-repo.git

2. **Install the dependencies from requirements.txt:**
## Data

This repository uses the AdultCensus, COMPAS, and Law School datasets. They can be found in the Datasets folder and are loaded using dataloader.py

## Reproducing Results
