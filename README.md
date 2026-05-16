<div align="center">
  <img src="https://github.com/user-attachments/assets/0544f664-cd2f-4847-b4fa-16097bdd9f1c" width="100%" alt="TRIP_U Cover">

  <h1>TRIP(U): Emotion-Based Personalized Travel Recommendation System</h1>

  <p><i>"The variable is you, the journey is your destination"</i></p>
  
  <br>

  [![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
  [![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-blue)](https://en.wikipedia.org/wiki/Natural_language_processing)
  [![Plotly](https://img.shields.io/badge/Visualization-Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)](https://plotly.com/)
</div>

---

## 🚀 Project Overview

**TRIP(U)** is an end-to-end intelligent recommendation engine that integrates Natural Language Processing (NLP) to introduce the emotional dimension into personalized travel planning. Built upon the core premise *"The variable is you, the journey is your destination"*, the system dynamically shifts and tailors travel destinations based on the user's specific psychological and emotional context rather than relying solely on static historical preferences.

---

## 🗺️ Methodology Roadmap: 8-Phase Architecture

The project is structured into **8 sequential phases** that cover the entire data lifecycle—ranging from affective computing and custom corpus annotation to cross-domain model validation and multi-criteria fusion scoring:

<div align="center">
  <img src="https://github.com/user-attachments/assets/e76151bb-4d2a-47b4-8845-c9ac97a74cea" width="100%" alt="TRIP_U 8-Phase Architecture Roadmap">
</div>

1. **Phase 1: Corpus Construction & Affective Computing** — Building a balanced emotional Gold Standard dataset.
2. **Phase 2: Core NLP Preprocessing Pipeline** — Text normalization, syntactic analysis, and domain masking.
3. **Phase 3: Supervised Modeling & Evaluation** — Feature vectorization and classifier benchmarking.
4. **Phase 4: Synthetic OOD Context Generation** — Designing a third normal form (3NF) Out-of-Distribution environment.
5. **Phase 5: Robustness & Cross-Domain Validation** — Testing model generalization capabilities outside the training domain.
6. **Phase 6: Multi-Criteria Recommendation Engine** — Blending emotional scores, thematic interests, behavioral profiles, and economic constraints.
7. **Phase 7: Ethical Guardrails & Advanced Analytics** — Implementing responsible AI mitigation filters and emotional dissonance tracking ($CCE$ / $CDE$).
8. **Phase 8: Scalability Roadmap & Transformers** — Future-proofing the architecture via Large Language Models ($RoBERTuito$) and dynamic clustering.

---

## 📊 Phase 1: Corpus Construction & Affective Computing

This phase establishes the foundational baseline of the project, migrating from raw text availability to a high-quality, non-biased emotional dataset.

### 1.1 Dataset Selection & Initial Filtering
* **Core Strategy:** The initial raw data was extracted from a Spanish movie review dataset on Kaggle. The cinematographic domain was strictly selected because film critiques provide highly expressive, syntax-rich, and emotionally fertile textual structures.
* **The Target:** Through a rigorous programmatic filtering and programmatic cleanup workflow, a definitive **Gold Standard corpus of 500 reviews** was isolated, ensuring a perfectly balanced distribution ($100$ items per emotional category).

<div align="center">
  <img src="https://github.com/user-attachments/assets/4ffba2ba-072c-4b23-a2bd-50bd44636738" width="100%" alt="Dataset Selection & Balance">
</div>

---

### 1.2 Affective Modeling: The Russell Circumplex Adaptation
* **Theoretical Framework:** Instead of treating emotions as arbitrary discrete labels, the system adapts **Russell’s Circumplex Model (1980)**. Affect is mapped across a continuous multi-dimensional space defined by two main axes: **Valence** (pleasure-displeasure) and **Arousal** (activation-energy).
* **The Categories:** Five emotional targets were isolated: *Sadness, Anxiety, Calm, Curiosity, and Joy*.
* **Mathematical Ordinality:** Although these represent discrete emotional states, they were intentionally modeled with mathematical ordinality ($0$ to $4$) within the pipeline. This geometric constraint allows the downstream loss functions and classification metrics to strictly penalize errors between distant emotional states (e.g., confusing *Sadness* with *Joy*) more severely than adjacent ones.

<div align="center">
  <img src="PASTE_IMAGE_1_2_URL_HERE" width="100%" alt="Russell Circumplex Model Mapping">
</div>

---

### 1.3 Double-Blind Annotation Pipeline
* **Bias Mitigation:** To completely eliminate homogeneous annotation bias, the entire corpus underwent a strict double-blind annotation methodology executed by two completely distinct socio-demographic profiles:
  * **Profile A (Technical/Analytical):** A Data Architect profile focusing on structural semantics.
  * **Profile B (Empathetic/Social):** A Costa Rican Social Educator profile focusing on empathetic and behavioral context.
* **Calibration Protocol:** The process was deployed across 3 sequential calibration batches ($50$ reviews each). After each batch, the annotation guidelines and codebooks were programmatically refined to resolve linguistic discrepancies and edge cases.

<div align="center">
  <img src="PASTE_IMAGE_1_3_URL_HERE" width="100%" alt="Double-Blind Annotation Workflow">
</div>

---

### 1.4 Inter-Annotator Agreement Metrics
To evaluate the absolute mathematical reliability and replicability of the manual tagging before training, three complementary agreement metrics were implemented:
1. **Exact Agreement Percentage:** Measuring crude overlap consistency.
2. **Quadratic Weighted Kappa ($\kappa_w$):** Acting as the primary validation metric, specifically configured to penalize distant emotional disagreements geometrically.
3. **Disagreement Distribution Analysis:** Mapping error density to detect systematic semantic deviations.

$$\kappa_w = 1 - \frac{\sum w_{ij} O_{ij}}{\sum w_{ij} E_{ij}}$$

* **Outcome:** Upon completing the calibration iterations, the agreement indices confidently surpassed the academic quality thresholds, formalizing the final annotation manual and cementing the perfectly balanced 500-review Gold Standard.

<div align="center">
  <img src="PASTE_IMAGE_1_4_URL_HERE" width="100%" alt="Agreement Metrics Evaluation">
</div>

---


---
