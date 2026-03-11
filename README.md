# Machine Learning-based Scientific Article Similarity Checking Research

[![Status](https://img.shields.io/badge/Status-Research--in--Progress-orange.svg)]()
[![Field](https://img.shields.io/badge/Field-Natural--Language--Processing-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

This repository contains the implementation of a **Hybrid Similarity Checking Model** for detecting semantic plagiarism in scientific articles. This work is part of a Master's Thesis in Informatics Engineering at Vilnius Gediminas Technical University.

## 🔬 Research Objective
The goal is to detect "intelligent plagiarism"—including paraphrasing, structural reordering, and LLM-generated content—that traditional lexical matching tools often fail to identify.

## 🏗️ Methodology (Thesis Part 1 Approved)
The system utilizes a **Hybrid SBERT-Structural Alignment Architecture**:
- **Semantic Layer:** Utilizes **Sentence-BERT (SBERT)** embeddings to capture underlying textual intent.
- **Structural Layer:** Extracts syntactic features (POS tag ratios, punctuation patterns, sentence morphology).
- **Alignment Layer:** Implements a **Semantic Smith-Waterman** dynamic programming algorithm to detect the "best path" of similarity through scrambled or reordered paragraphs.

## 🚀 Implementation Status
- [x] **Core Engine:** Hybrid scoring with weighted Semantic/Structural components.
- [x] **SOTA Alignment:** Semantic Smith-Waterman sequence alignment.
- [x] **Evaluation Pipeline:** character-level Precision, Recall, and F1-score tracking.
- [x] **Mock Data Support:** Built-in generator for PAN-style dataset simulation.
- [ ] **Current Milestone:** Validation and hyperparameter tuning on the official **PAN 2025 Dataset**.

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/Utshox/Machine-Learning-based-Scientific-Article-Similarity-Checking-Research.git

# Setup Virtual Environment
python3 -m venv venv
source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
```

## 📊 Quick Start
To run the current evaluation on mock data:
```bash
python run_evaluation.py
```

## 📜 Continuous Tracking
Progress and scientific verdicts are tracked in:
- `PROGRESS.md`: Detailed changelog and task tracking.
- `VERDICTS.md`: Technical audits of alignment with the original thesis proposal.

---
**Note:** This is an ongoing research project. Large-scale dataset evaluation results are forthcoming.
