# REAL: Relation Extraction and Linking in one go

**REAL** (*Relation Extraction And Linking*) is an end-to-end autoregressive framework designed to perform **Relation Extraction (RE)** and **Entity Linking (EL)** simultaneously within a single generation step.

This project builds upon the **REBEL** architecture (BART-large) by integrating disambiguation mechanisms directly into the sequence generation, mapping extracted entities to unique Knowledge Graph (KG) identifiers (e.g., Wikidata IDs) in real-time.

## 🚀 Key Concepts

Traditional Information Extraction often relies on multi-step pipelines (NER → EL → RE) where errors cascade through each stage. **REAL** bypasses this by:
* **Joint Learning:** Using a Seq2Seq approach to extract relations and link entities to a KG in one pass.
* **Enriched Linearization:** Replacing generic markers with specific tokens that represent KG identifiers.
* **Hallucination Reduction:** Grounding the language model's output in a structured reference graph.

## 🧬 Methodology

### Triplet Linearization
The model is trained to transform raw text into a linearized string of triplets including unique identifiers:
`[triplet] Subject [ID_Subj] Relation [ID_Rel] Object [ID_Obj]`

### Training Pipeline
1. **Pre-training:** Leveraging the REBEL dataset to stabilize the relation extraction capabilities.
2. **Fine-tuning:** Specialized adaptation on **DocRED-IE** and **DWIE** datasets for document-level reasoning and precise entity linking.

## 🛠️ Technical Stack

* **Frameworks:** PyTorch Lightning (Training lifecycle) & Hugging Face Transformers
* **Configuration:** Hydra & OmegaConf (Hierarchical configuration management)
* **Infrastructure:** Slurm scripts for distributed training on clusters
* **Logging:** TensorBoard

## 📁 Project Structure

* `requirements.txt`: Python dependencies.
* `scripts/`: Contains `.sh` files for cluster job submission and training routines.
* `conf/`: Hydra configuration files for models and datasets.
* `src/`: Core logic for model architecture, data loading, and linearization.

## 📝 Background

This repository contains the implementation developed during my **Master’s Thesis at Paris-Saclay University**. The work was hosted at the **LISN** (Laboratoire Interdisciplinaire des Sciences du Numérique) within the **LaHDAK** team.

**Author:** Julien Rolland  
**Supervisors:** Fatiha Saïs & Nicoleta Preda
