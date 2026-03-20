# Interpretable Deep Reinforcement Learning with Differentiable Decision Trees for Cost-Effective Medical Diagnosis

This repository contains the official PyTorch implementation of the paper **"Interpretable Deep Reinforcement Learning with Differentiable Decision Trees for Cost-Effective Medical Diagnosis"**.

## 📋 Overview

Medical diagnosis requires balancing the high accuracy of predictive models with the economic costs of clinical tests. Traditional Deep Reinforcement Learning (DRL) methods, while effective, often act as "black boxes," limiting their adoption in safety-critical healthcare settings.

This project introduces **Differentiable Decision Tree Policy Optimization (DDT-PO)**, a novel framework that integrates **Differentiable Decision Trees (DDT)** into the **Semi-Model-based Deep Diagnosis Policy Optimization (SM-DDPO)** architecture.

**Key Contributions:**

* **Interpretable Policy:** Replaces opaque neural networks with Differentiable Decision Trees, allowing the learned policy to be visualized as explicit, human-readable *If-Then* rules.
* **Cost-Effective:** Achieves a **50% reduction in average diagnostic costs** compared to full-feature baselines while maintaining comparable F1 scores.
* **Pareto Optimization:** Recovers a proper Pareto frontier of cost-performance trade-offs, allowing clinicians to adjust sensitivity based on budget constraints.



## 📂 Repository Structure

```text
.
├── baselines.py                          # Baseline evaluation utilities
├── train.py                              # Single-run baseline SM-DDPO training
├── train_ddt.py                          # Single-run DDT-SM-DDPO training
├── multi_train.py                        # Multi-run baseline training
├── multi_train_ddt.py                    # Multi-run DDT training
├── multi_train_hybrid.py                 # Multi-run hybrid training (NN + DDT variants)
├── multi_train_ddt_depth_sweep.py        # DDT depth sweep experiments
├── classifiers/
│   ├── classifier.py                     # Baseline neural classifier
│   └── classifier_ddt.py                 # Differentiable Decision Tree classifier
├── data_preprocessing/
│   ├── data_loader.py                    # Dataset loading utilities
│   └── blood_panel_data_preprocessing.py # Blood panel preprocessing
├── ddt_examples/
│   ├── ddt_classifier.py                 # Standalone DDT classifier example
│   └── ddt_rl_example.py                 # DDT reinforcement-learning example
├── helpers/
│   ├── meta_train.py                     # Meta-training helpers
│   ├── my_result_writer.py               # Logging/result writing helpers
│   └── util.py                           # Shared utility functions
├── imputer/
│   ├── flow_models.py                    # Flow model components
│   ├── imputation.py                     # Missing-value imputation pipeline
│   └── nflow.py                          # Normalizing flow implementation helpers
├── inference/
│   ├── auto_explain_model.py             # Rule extraction / model explanation
│   ├── get_graphs.py                     # Plotting and graph generation
│   └── statistical_significance.py       # Statistical significance analysis
├── rl_agent/
│   ├── rl.py                             # Baseline RL agent
│   └── rl_ddt.py                         # Interpretable DDT-based RL agent
├── requirements.txt
└── README.md
```

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/sumedh-ramteke/interpretable-rl.git
cd interpretable-rl

```


2. **Install dependencies:**
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```


## 🏃 Usage

### 1. Train the Interpretable Agent (Ours)

To train the proposed model using Differentiable Decision Trees for both the classifier and the RL policy:

```bash
python train_ddt.py
```

For robust results, use the multi-run training script:

```bash
python multi_train_ddt.py
```


* **Configuration:** You can adjust the penalty ratio (`lambda`) and tree depth inside the script to explore different points on the Pareto frontier.
* **Output:** Checkpoints and logs will be saved to `./training_results/results_ddt`.

### 2. Train the Baseline (SM-DDPO)

To train the standard Semi-Model-based Deep Diagnosis Policy Optimization (using Neural Networks):

```bash
python train.py

```

Similarly, the robust approach for multi-run training script:

```bash
python  multi_train.py

```

### 3. Hybrid Multi-Run Training

To run hybrid experiments that combine baseline and DDT-style components:

```bash
python multi_train_hybrid.py
```

### 4. DDT Depth Sweep

To run depth ablation/sweep studies for Differentiable Decision Trees:

```bash
python multi_train_ddt_depth_sweep.py
```

### 5. Visualization

After training, the soft decision trees can be discretized into hard rules for clinical verification. The extraction logic is located in `inference/auto_explain_model.py`.

## 📄 References

If you find this repository useful, please cite our paper as well as the foundational works on SM-DDPO and DDTs:

### Primary Citation (This Work)

```bibtex
@article{ramteke2026interpretable,
  title={Interpretable Deep Reinforcement Learning with Differentiable Decision Trees for Cost-Effective Medical Diagnosis},
  author={Ramteke, Sumedh and Deshpande, Umesh A.},
  journal={Under Review},
  year={2026}
}
```

### Foundational Works

This repository builds upon the following excellent research:

**1. SM-DDPO Framework**
Zheng Yu, Yikuan Li, Yuan Luo, Joseph C. Kim, Kaixuan Huang, Mengdi Wang. "Deep Reinforcement Learning for Cost-Effective Medical Diagnosis." ICLR 2023. 

```bibtex
@inproceedings{yu2023deep,
  title={Deep Reinforcement Learning for Cost-Effective Medical Diagnosis},
  author={Yu, Zheng and Li, Yikuan and Luo, Yuan and Kim, Joseph C and Huang, Kaixuan and Wang, Mengdi},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

```

**2. Differentiable Decision Trees**
Andrew Silva, Taylor Killian, Ivan Rodriguez Jimenez, Sung-Hyun Son, Matthew Gombolay. "Optimization Methods for Interpretable Differentiable Decision Trees in Reinforcement Learning." AISTATS 2020. 

```bibtex
@inproceedings{silva2020optimization,
  title={Optimization Methods for Interpretable Differentiable Decision Trees in Reinforcement Learning},
  author={Silva, Andrew and Killian, Taylor and Jimenez, Ivan Rodriguez and Son, Sung-Hyun and Gombolay, Matthew},
  booktitle={Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics},
  year={2020},
  publisher={PMLR}
}

```

## 🤝 Acknowledgements

This codebase is a development repository for interpretable medical diagnosis. It is derived from the implementation of SM-DDPO by Yu et al. (https://github.com/Zheng321/Deep-Reinforcement-Learning-for-Cost-Effective-Medical-Diagnosis). We acknowledge their contribution to the field of cost-effective medical diagnosis. Please refer to their original work for specific implementation details regarding the environment and reward shaping mechanisms.
