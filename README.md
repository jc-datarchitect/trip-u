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

## Methodology Roadmap: 8-Phase Architecture

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

## Phase 1: Corpus Construction & Affective Computing

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

## Phase 2: Core NLP Preprocessing Pipeline

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

## Phase 3: Supervised Modeling & Evaluation Benchmarking

This phase focuses on translating the normalized textual corpus into mathematical features to train, benchmark, and contrast supervised classification algorithms, isolating the optimal predictive engine for emotional multi-class detection.

### 3.1 Feature Extraction & Vectorization Strategy
* **The Approaches:** Two classical text vectorization paradigms were systematically evaluated: **Bag of Words (BoW / CountVectorizer)** and **TF-IDF (Term Frequency-Inverse Document Frequency)**.
* **Controlled Tuning:** To prevent feature explosion and ensure structural parity, both pipelines were constrained to a maximum of $2,500$ features, an $ngram\_range=(1, 2)$ (capturing unigrams and bigrams), and a $min\_df=2$ threshold to filter out rare anomalies.
* **Sparsity Mapping:** The final document-term matrices yielded a high sparsity profile ($99.0\%$), a standard mathematical footprint in natural language modeling that requires robust linear or probabilistic separating boundaries.

<div align="center">
  <img src="https://github.com/user-attachments/assets/0e936495-4a41-4959-9a59-5c399b32fa36" width="100%" alt="Feature Vectorization Strategy">
</div>

---

### 3.2 Proposed Full-Factorial Architecture Layout
* **Experimental Matrix:** To isolate the most reliable classification engine, both vectorization strategies were crossed with two distinct supervised learning algorithm families: **Multinomial Naive Bayes (NB)** (probabilistic baseline) and **Linear Support Vector Classifier (Linear SVC / SVM)** (geometric margin optimization).
* **The Four Evaluated Configurations:**
  1. $BoW + Multinomial\ Naive\ Bayes\ (NB)$
  2. $TF\text{-}IDF + Multinomial\ Naive\ Bayes\ (NB)$
  3. $BoW + Linear\ SVC\ (SVM)$
  4. $TF\text{-}IDF + Linear\ SVC\ (SVM)$

<div align="center">
  <img src="https://github.com/user-attachments/assets/1815a1ab-3c73-44f1-9633-d0339fba6f85" width="100%" alt="Proposed Experimental Pipelines">
</div>

---

### 3.3 Macro Performance Metrics Comparison
* **Overall Evaluation:** All four candidate models achieved highly consistent macro performance scores across precision, recall, and F1 metrics, fluctuating around the $0.77$ to $0.83$ range. This performance level statistically confirms the semantic cleanlines and balanced nature of the initial Gold Standard corpus.
* **Top Performers:** The probabilistic baseline ($BoW + NB$) and the geometric margin classifier ($TF\text{-}IDF + SVM$) emerged as the frontrunners, showing near parity at a macro evaluation level. This proximity required a deeper dive into class-specific performance.

<div align="center">
  <img src="https://github.com/user-attachments/assets/be5e1035-0480-4267-9f91-49ef4a566ef4" width="100%" alt="Macro Performance Comparison">
</div>

---

### 3.4 Multi-Class ROC Curves & Area Under the Curve (AUC) Analysis
* **Discriminative Capacity:** Looking at the Receiver Operating Characteristic (ROC) curves, the true positive rate versus false positive rate was mapped across all five target categories (*Tristeza, Ansiedad, Calma, Curiosidad, Alegría*).
* **Key Observations:** The $BoW + NB$ setup achieved exceptional stability, maintaining uniform AUC scores between $0.95$ and $0.97$ for all emotions. Conversely, the SVM alternatives exposed a performance degradation in high-activation positive states, dropping to an AUC of $0.84$ for *Alegría*.

<div align="center">
  <img src="https://github.com/user-attachments/assets/23013a67-77bf-4cbf-8ff3-357eaefd2ebd" width="100%" alt="Multi-Class ROC Curves Comparison">
</div>

---

### 3.5 Precision-Recall (P-R) Curves Analysis
* **Imbalance & Class Focus:** Because macro metrics can sometimes hide boundary weaknesses, class-specific Precision-Recall curves were computed to inspect the True Positive Rate against the Positive Predictive Value across changing thresholds.
* **Granular Reliability:** The Average Precision (AP) scores validate that $BoW + NB$ handles subtle emotional transitions with superior accuracy. It maintains remarkably tight envelopes for complex classes like *Anxiety* ($AP = 0.87$) and *Curiosity* ($AP = 0.94$), outperforming the SVM pipelines which showed broader precision drops.

<div align="center">
  <img src="https://github.com/user-attachments/assets/3fb839bf-45d7-420c-a23d-380fe0f9e0b2" width="100%" alt="Precision-Recall Curves Comparison">
</div>

---

### 3.6 Multi-Class Confusion Matrix Diagnosis
* **Error Distribution:** The multi-class confusion matrices display a highly prominent, well-defined main diagonal across all candidate setups, indicating highly accurate categorical assignments.
* **Proximity Boundaries:** Most misclassifications are strictly confined to adjacent coordinates within the underlying psychological emotional space. The $BoW + NB$ pipeline minimized severe cross-spatial contradictions (e.g., confusing low-activation negative states with high-activation positive states), maintaining a clean error pattern.

<div align="center">
  <img src="https://github.com/user-attachments/assets/9a645aff-a684-411d-a54e-81db9531ba60" width="100%" alt="Confusion Matrices Benchmarking">
</div>

---

### 3.7 Definitive Production Architecture Selection
* **The Selection:** **Bag of Words + Multinomial Naive Bayes (BoW + NB)** was officially designated as the production classifier engine.
* **Justification Portfolio:** While macro metrics showed competitive numbers across the board, the $BoW + NB$ model was selected based on four core technical criteria:
  * **Solid Macro Metrics:** Highly reliable macro precision, recall, and F1 foundations.
  * **P-R Curve Consistency:** Superior Average Precision balances, even when handling complex, overlapping emotional boundaries.
  * **Optimized Error Envelopes:** Cleaner confusion matrix diagonals with minimal cross-category extreme errors.
  * **Uniform ROC Ensembles:** Highest and most uniform multi-class AUC metrics across all 5 emotional dimensions.

<div align="center">
  <img src="https://github.com/user-attachments/assets/37d81735-835d-4bd7-a8d3-6c81bf9e043c" width="100%" alt="Final Model Selection Criteria">
</div>

---

## Phase 4: Synthetic OOD Context & Relational Database Architecture

This phase details the design and engineering of an Out-of-Domain (OOD) testing environment, validating the model's generalization boundaries while establishing a structured data pipeline.

### 4.1 Out-of-Domain (OOD) Rationalization & Synthetic User Cohort
* **The Domain Shift Challenge:** While the emotional classification backbone was trained on film reviews, the production recommendation engine is designed to parse raw social media streams. To guarantee robust **domain generalization**, evaluating the model outside its training domain was computationally mandatory.
* **The Privacy & Linking Bottleneck:** Real-world social media datasets often enforce anonymity or lack linked user behavioral profiles (budgets, demographic backgrounds, static interests). 
* **The Solution:** A controlled, high-fidelity synthetic OOD corpus was architected. It features **25 heterogeneous user personas** with diverse nationalities, age brackets, travel budgets, and thematic interests. Each persona generates distinct, unstructured text streams within a realistic environment, serving as the stress-test baseline for the recommendation engine.

<div align="center">
  <img src="https://github.com/user-attachments/assets/dcde64a8-5f66-4767-b6fd-07ff31c4667d" width="100%" alt="OOD Corpus Justification and Synthetic Cohort">
</div>

---

### 4.2 Relational Schema & Third Normal Form (3NF) Database Normalization
* **Data Architecture:** To scale the multi-criteria recommendation engine, the synthetic OOD ecosystem was structured into a fully normalized **Relational Database Management System (RDBMS)**.
* **Normalization Protocol:** The schema was meticulously optimized up to the **Third Normal Form (3NF)** to ensure transactional integrity, eliminate data redundancy, and prevent anomalies.
* **Entity Relationship Layout:** Isolated relational entities—including *Users, Travel Destinations, Personal Interests, and Transport/Living Costs*—were mapped using strict Primary Key (PK) and Foreign Key (FK) constraints, establishing the data foundation for downstream recommendation filtering.

<div align="center">
  <img src="https://github.com/user-attachments/assets/5c0cb87e-4e75-461f-822f-c3c01a260f7c" width="100%" alt="3NF Database Normalization and Schema Architecture">
</div>

---

## Phase 5: Robustness & Cross-Domain Validation (ID vs. OOD)

This phase presents the definitive validation stress-test of the selected $BoW + NB$ production model, contrasting its performance between the controlled In-Domain (ID) cinema setting and the novel Out-of-Domain (OOD) unstructured social context.

### 5.1 In-Domain (ID) vs. Out-of-Domain (OOD) Performance Degradation
* **The Domain Shift Expected Drop:** Subjecting the pipeline to the synthetic OOD corpus triggered an expected performance compression when contrasted against the near-perfect metrics achieved during movie review cross-validation.
* **The Baseline Victory:** Despite the domain shift, the macro evaluation metrics remain substantially and confidently above the **$20\%$ random baseline probability** dictated by the 5-class classification framework. This mathematical margin confirms that the entity masking protocol successfully forced the architecture to learn transferable, domain-independent affective syntax structures.

<div align="center">
  <img src="https://github.com/user-attachments/assets/dc4edbdf-565e-497f-bfb7-f625db3bf0e2" width="100%" alt="In-Domain vs Out-of-Domain General Comparison">
</div>

---

### 5.2 Out-of-Domain ROC and Precision-Recall Curve Inferences
* **ROC AUC Boundaries:** Looking closer at the OOD validation graphs, the model exhibits high structural resilience. The Receiver Operating Characteristic (ROC) plots reveal stable, solid classification boundaries across all vectors, yielding competitive Area Under the Curve (AUC) metrics ranging from $0.82$ up to $0.89$. 
* **Precision-Recall & Class Friction:** The class-specific Precision-Recall curves expose how the model behaves under non-cinematographic constraints. High-activation positive states like *Alegría* represent the highest classification friction ($AUC = 0.82$ / $AP = 0.59$), while abstract or high-arousal states like *Curiosidad* ($AUC = 0.89$) and *Ansiedad* ($AUC = 0.87$) retain high mathematical focus.

<div align="center">
  <img src="https://github.com/user-attachments/assets/8c73cced-7c8c-42ef-8aca-ca0f986a1897" width="100%" alt="Out-of-Domain ROC and Precision-Recall Curves Validation">
</div>

---

### 5.3 Out-of-Domain Multi-Class Confusion Matrix Mapping
* **Diagonal Dominurally Maintained:** The OOD multi-class confusion matrix solidifies the structural viability of the machine learning classifier. The primary diagonal confidently retains its dominant density distribution under stress.
* **Proximity Concentrated Variance:** Classification errors do not disperse randomly across the matrix; instead, they cluster tightly inside adjacent emotional boundaries within the adaptive Russell affect matrix. This validation ensures that the output layer provides a reliable, logical foundation for the multi-criteria recommendation loops.

<div align="center">
  <img src="https://github.com/user-attachments/assets/143b1a32-aef4-433d-836d-544b943ab3e6" width="100%" alt="Out-of-Domain Multi-Class Confusion Matrix">
</div>

---

