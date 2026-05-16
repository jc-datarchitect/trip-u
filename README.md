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
  <img src="https://github.com/user-attachments/assets/99c36e38-252e-4caf-a958-cfeab31ce659" width="100%" alt="Russell Circumplex Model Mapping">
</div>

---

### 1.3 Double-Blind Annotation Pipeline
* **Bias Mitigation:** To completely eliminate homogeneous annotation bias, the entire corpus underwent a strict double-blind annotation methodology executed by two completely distinct socio-demographic profiles:
  * **Profile A (Technical/Analytical):** A Data Architect profile focusing on structural semantics.
  * **Profile B (Empathetic/Social):** A Costa Rican Social Educator profile focusing on empathetic and behavioral context.
* **Calibration Protocol:** The process was deployed across 3 sequential calibration batches ($50$ reviews each). After each batch, the annotation guidelines and codebooks were programmatically refined to resolve linguistic discrepancies and edge cases.

<div align="center">
  <img src="https://github.com/user-attachments/assets/fedd6c95-e021-4048-9980-5f890d8da5b5" width="100%" alt="Double-Blind Annotation Workflow">
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
  <img src="https://github.com/user-attachments/assets/71cda5a2-48cf-4045-ba70-2db205b36f4d" width="100%" alt="Agreement Metrics Evaluation">
</div>

---

## 🧪 Phase 2: Core NLP Preprocessing Pipeline

Once the Gold Standard corpus was consolidated, a specialized text normalization and lexical analysis pipeline was deployed. The main engineering challenge was stripping domain-specific noise while strictly preserving the underlying affective and emotional signals.

### 2.1 Preprocessing Pipeline & Text Normalization
* **The Process:** The raw textual data underwent a structured normalization workflow consisting of case folding (lowercasing), selective accent removal to preserve emotional emphasis, punctuation filtering, and custom tokenization.
* **Objective:** This pipeline transforms noisy, unstructured review text into clean, modelable features, ensuring that the feature space reflects true semantic content without altering the original affective weight.

<div align="center">
  <img src="https://github.com/user-attachments/assets/bd97526a-72fa-4a3a-85e3-75ad114fa22c" width="100%" alt="NLP Normalization and Preprocessing Pipeline">
</div>

---

### 2.2 Named Entity Masking (Critical Feature Engineering)
* **The Overfitting Risk:** Since the model was trained on movie reviews, it faced a high risk of overfitting to domain-specific jargon, director names, actor names, and movie titles.
* **The Solution:** A programmatic **Named Entity Masking** filter was designed. Cinema-specific entities and technical vocabulary were systematically replaced with neutral placeholder tags.
* **Cross-Domain Adaptability:** This constraint forced the classification algorithms to learn abstract syntactic and emotional structures rather than thematic keywords. This domain independence is the architectural key that allows the model to successfully transfer its predictive power to raw social media texts later on.

<div align="center">
  <img src="https://github.com/user-attachments/assets/80479f98-49fd-4652-a15c-49fc575781c8" width="100%" alt="Named Entity Masking Strategy">
</div>

---

### 2.3 Lexical Feature Check: Top 20 Words per Emotion
* **Sanity Validation:** An exploratory frequency distribution analysis was conducted to isolate the top 20 most frequent terms for each target emotion.
* **Outcome:** The dominant keywords inside each cluster showed total semantic coherence with their corresponding psychological states. This mathematically verified that the aggressive preprocessing and entity masking pipeline did not distort or weaken the underlying emotional signal.

<div align="center">
  <img src="https://github.com/user-attachments/assets/4b1a2717-3ced-45b5-8940-af431de63064" width="100%" alt="Top 20 Emotional Keywords Distribution">
</div>

---

### 2.4 Morphosyntactic Distribution Across Affective Spaces (POS-Tagging)
* **Syntactic Footprints:** A Part-of-Speech (POS) tagging analysis revealed that different emotional states fundamentally alter the grammatical composition of a sentence.
* **Key Discovery:** *Sadness* exhibits a highly static behavior, characterized by higher noun density and fewer action verbs. Conversely, *Curiosity* triggers a sharp increase in verbal structures, directly matching a high psychological arousal (activation). This proves that emotional states do not just dictate vocabulary, but actively shape syntax architecture.

<div align="center">
  <img src="https://github.com/user-attachments/assets/bb9609aa-077c-43c7-95e3-65c0f897d309" width="100%" alt="Grammatical POS Category Distribution per Emotion">
</div>

---

### 2.5 Lexical Diversity and Vocabulary Richness Analysis
* **Metrics:** The Type-Token Ratio (TTR) and lexical richness indices were computed for each emotional sub-corpus to evaluate the structural complexity of vocabulary usage.
* **Arousal Correlation:** The analysis demonstrates a clear correlation between psychological activation and vocabulary expanse. High-arousal states like *Curiosity* or *Joy* yield a wider, more creative deployment of unique lemmas, whereas low-activation or highly restrictive states like *Sadness* exhibit linguistic repetition and a more concentrated, localized vocabulary footprint.

<div align="center">
  <img src="https://github.com/user-attachments/assets/5fc73bde-8103-4b5b-951a-b1f1351afdc3" width="100%" alt="Lexical Diversity and Type-Token Ratio Matrix">
</div>

---


---
