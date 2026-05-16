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
  <img src="https://github.com/user-attachments/assets/1815a1ab-3c73-44f1-9633-d0339fba6f85" width="70%" alt="Proposed Experimental Pipelines">
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
  <img src="https://github.com/user-attachments/assets/37d81735-835d-4bd7-a8d3-6c81bf9e043c" width="80%" alt="Final Model Selection Criteria">
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
  <img src="https://github.com/user-attachments/assets/143b1a32-aef4-433d-836d-544b943ab3e6" width="70%" alt="Out-of-Domain Multi-Class Confusion Matrix">
</div>

---

## Phase 6: Multi-Criteria Recommendation Engine & Algorithmic Fusion

This phase outlines the core architectural framework of the recommendation engine, migrating from emotional intelligence to an operational, multi-criteria decision-making system.

### 6.1 Multi-Criteria System Architecture Overview
* **The Structural Blueprint:** The system is engineered around a multi-criteria decision analysis (MCDA) framework that integrates four independent, specialized pillars: *Emotion, Thematic Interests, Behavioral Traveler Profiles, and Financial Constraints*.
* **Dynamic vs. Static Balance:** The **Emotional Score** acts as the core dynamic vector, capturing shifting, real-time context. The remaining three pillars provide structural stabilization, anchoring the engine to the user's permanent preferences and hard limitations.

<div align="center">
  <img src="https://github.com/user-attachments/assets/4f39cb13-314c-45de-ae51-3db3d525097b" width="100%" alt="Multi-Criteria Recommendation System Architecture">
</div>

---

### 6.2 Pillar 1: Dual Emotional Scoring Matrix
* **Signal Cross-Validation:** To mitigate alignment anomalies and subjective biases, the Emotional Score fuses two independent signals:
  1. **Self-Perceived Emotion:** Manually declared by the user via the frontend interface.
  2. **Inferred Emotion:** Computationally extracted from the user's raw text using the production $BoW + NB$ classifier.
* **Destination Mapping:** By mathematically crossing this joint input vector against the structured emotional matrix assigned to each target destination, the engine derives a highly resilient and validated emotional affinity score.

<div align="center">
  <img src="https://github.com/user-attachments/assets/d1bab853-2bde-47db-a896-5e4a43974d06" width="100%" alt="Pillar 1: Emotional Score Signal Fusion">
</div>

---

### 6.3 Pillar 2: Thematic Interest & Affinity Stabilization
* **Semantic Overlap:** This component evaluates categorical affinity by calculating the overlap between the user's explicitly declared personal interests (e.g., trekking, gastronomy, architecture) and the thematic tags mapped to each destination.
* **The Scoring Scale:** Based on the intersection density of identical attributes, a discrete score ranging from $0\text{ to }5$ is computed. This metric acts as a structural anchor, ensuring that the engine's outputs remain intrinsically relevant to the user's baseline lifestyle preferences.

<div align="center">
  <img src="https://github.com/user-attachments/assets/7898922f-018d-43f4-bfd8-00da20f681ef" width="100%" alt="Pillar 2: Thematic Interest Affinity">
</div>

---

### 6.4 Pillar 3: Behavioral Traveler Profile Mapping
* **Behavioral Typology:** The system introduces an algorithmic behavioral tracking layer that programmatically classifies users into definitive traveler profiles (e.g., *Explorer, Relaxed, Socializer, Cultural*).
* **Cross-Referencing Matrices:** Every travel destination within the relational database holds a predefined vector of suitability weights mapped against these exact behavioral typologies. The engine executes a matrix dot product between the user's inferred profile and the destination vectors to calculate behavioral alignment.

<div align="center">
  <img src="https://github.com/user-attachments/assets/b58a822a-7a98-4218-be10-58768c449d63" width="100%" alt="Pillar 3: Behavioral Traveler Profiles">
</div>

---

### 6.5 Pillar 4: Economic Constraints & Compound Cost Index
* **Financial Indexing:** The economic filter computes a customized **Compound Cost Index ($CCI$)** for every potential travel route. This index dynamically balances and weights two high-impact financial variables:
  
  $$CCI = w_1 \cdot \text{Cost of Living} + w_2 \cdot \text{Transport Costs}$$
  
* **Tiered Bracket Evaluation:** The resulting compound index is mapped against the user's strict financial threshold bracket, generating a localized score that guarantees recommendations are perfectly calibrated within realistic budget parameters.

<div align="center">
  <img src="https://github.com/user-attachments/assets/d3a07dea-9d45-4543-8a17-8b53438c4ae8" width="100%" alt="Pillar 4: Economic Constraints and Compound Cost">
</div>

---

### 6.6 Algorithmic Fusion & Personalized Ranking Integration
* **Normalization & Weighting:** To merge these highly heterogeneous scales into a singular, unified scoring layer, each pillar undergoes a normalization pipeline. Weights are allocated strategically, establishing the **Emotional Vector as the algorithmic heart with a dominant $45\%$ weight distribution**.
* **The Scoring Engine:** The ultimate global score ($GS$) for each individual destination ($d$) and user ($u$) is calculated using a linear combination framework:

  $$GS(d,u) = 0.45 \cdot S_{\text{Emotion}} + w_i \cdot S_{\text{Interests}} + w_j \cdot S_{\text{Behavior}} + w_k \cdot S_{\text{Economic}}$$

* **Final Rank Generation:** The recommendation matrix is sorted descendingly per user, isolating the destination that achieves the highest global score as the absolute personalized travel choice.

<div align="center">
  <img src="https://github.com/user-attachments/assets/dc99dc9e-53d7-4561-857e-72716855c523" width="100%" alt="Algorithmic Weight Fusion and Ranking Integration">
</div>

---

## Phase 7: Responsible AI, Advanced Behavioral Analytics & Geographic Inferences

This final phase integrates production-grade ethical guardrails, custom affective monitoring coefficients, and localized macroeconomic data visualizations extracted from the system's runtime analytics.

### 7.1 Ethical Guardrails & Responsible AI Postulates (Hard vs. Soft Filters)
* **Algorithmic Accountability:** To ensure compliance with modern Responsible AI guidelines, a specialized multi-tier ethical validation layer was embedded directly into the downstream routing logic.
* **The Filter Taxonomy:**
  * **Hard Exclusion Filters:** Strict logical constraints preventing the engine from recommending a destination that matches the user's current geographic location ($Origin \neq Destination$).
  * **Soft Informative Filters:** Non-paternalistic, friction-free advisory banners triggered under high-sensitivity configurations:
    * *Persistent Sadness Alert:* Triggered when consecutive text analysis iterations indicate prolonged low-activation states.
    * *LGBTQ+ Protection Inferences:* Informing users of localized legal limitations in non-inclusive destinations.
    * *Accessibility Warning Vector:* Flagging destinations displaying prominent infrastructure barriers for users with documented mobility constraints.

<div align="center">
  <img src="https://github.com/user-attachments/assets/3b86e3af-9a01-4a3c-8df2-b682eea723d6" width="100%" alt="Ethical Filters Framework">
</div>

---

### 7.2 Individual Emotional Evolution (Custom Coefficients: CCE & CDE)
* **Dynamic Time-Series Monitoring:** The platform evaluates user behavior across seasonal timelines, tracking the interplay between User Self-Perception and Textual Model Inferences.
* **Proprietary Mathematical Framework:** Two distinct custom behavioral coefficients were engineered:
  * **Emotional Consistency Coefficient ($CCE$):** Quantifies the variance and structural stability of the user's self-perceived baseline over time (represented by the explicit red timeline).
  * **Emotional Dissonance Coefficient ($CDE$):** Mathematically computes the integrated area under the curve representing the absolute error or divergence between self-perceived states and the model's textual inference layer (represented by the shaded gray geometric boundaries).

<div align="center">
  <img src="https://github.com/user-attachments/assets/b6b93694-3d69-4164-8f10-fa745ed3783d" width="100%" alt="Individual Affective Timelines">
</div>

---

### 7.3 Multi-Quadrant Affective Profile Mapping
* **The Behavioral Matrix:** By crossing the continuous dimensions of the $CCE$ and $CDE$ indices, users are programmatically mapped into a specialized four-quadrant coordinate space.
* **Typology Stratification:** This multi-criteria mapping splits user behavior into four core optimization targets, enabling the engine to adapt its core mathematical recommendation weights ($w_i$) based on the user's strategic quadrant profile.

<div align="center">
  <img src="https://github.com/user-attachments/assets/b83aee49-8f60-4892-b666-66e752a9855f" width="100%" alt="MCDA Affective Matrix Mapping">
</div>

---

### 7.4 Cohort Distribution & Predictive Friction Analysis
* **Statistical Volume Analysis:** Evaluating the 25-user heterogeneous cohort reveals that **$56\%$ of users reside inside the Consistent-Sonant quadrant ("El usuario lúcido")**, confirming high system stability.
* **Predictive Friction:** The remaining population highlights complex behavioral exceptions, isolating highly volatile or masked profiles ("El usuario volátil" / "El usuario enmascarado") that require specialized textual inference overrides due to high self-reporting bias.

<div align="center">
  <img src="https://github.com/user-attachments/assets/43b077f6-4454-4ed7-8ff7-b35d9d9532a4" width="100%" alt="Cohort Quadrant Distributions">
</div>

---

### 7.5 Statistical Density Profiling (CCE vs. CDE Violin Plots)
* **Kernel Density Distribution:** Violin plots crossing the full cohort distributions reveal a multi-modal distribution inside the $CCE$ boundary, showcasing a high concentration around the $0.65$ efficiency mark.
* **Dissonance Control:** The $CDE$ distribution exhibits a highly constrained interquartile range ($IQR$), confirming that the model's textual inferences maintain a tight, controlled error envelope ($< 0.40$) across the vast majority of out-of-domain interactions.

<div align="center">
  <img src="https://github.com/user-attachments/assets/be6dc876-02ee-4a97-973a-2b57e028a3ab" width="100%" alt="Coefficient Distribution Densities">
</div>

---

### 7.6 Macro-Seasonal Affective Heatmap Analysis
* **Macro-Environmental Signals:** The system aggregates full transaction streams into seasonal heatmaps, comparing declared self-perception matrices against raw inferred textual trends.
* **Key Insights:** While self-perceived metrics show relatively flat distributions throughout the year, the textual model successfully captures micro-trends, such as a prominent spike in high-activation anxiety metrics ($0.29$) during the spring window and low-arousal calm states ($0.27$) during the autumn/winter transitions.

<div align="center">
  <img src="https://github.com/user-attachments/assets/c283519b-b3b9-4279-8d51-ce393c62e5cc" width="100%" alt="Seasonal Affective Heatmaps">
</div>

---

### 7.7 Multi-Demographic Radar Trajectory Intersections
* **Cross-Attribute Interactions:** Multi-axis radar visualizations map the geometric intersections between emotional states, behavioral traveler traits, financial budgets, and distinct age cohorts.
* **Correlative Features:** The radar layers isolate high-value behavioral insights, mapping how low-budget brackets heavily intersect with high-arousal exploratory interest patterns, whereas older demographics or higher-budget profiles display tight alignment with low-activation, high-comfort destination targets.

<div align="center">
  <img src="https://github.com/user-attachments/assets/4e58ea7b-0f46-4d1c-a3e2-0aa6c3757a2e" width="100%" alt="Demographic Radar Intersections">
</div>

---

### 7.8 Global Routing Geo-Spatial Visualization
* **The Final Routing Map:** The operational output of the global multi-criteria recommendation loops is mapped using spatial geolocated vectors, connecting user origin nodes (blue markers) to their optimal, customized travel destinations (red target nodes).
* **Validation Confirmation:** The geographic paths validate the system's complex cross-criteria constraints, confirming that the engine successfully skips nearby sub-optimal choices to find international destinations that best align with the user's emotional and budget profile.

<div align="center">
  <img src="https://github.com/user-attachments/assets/45b9e3cb-ad6e-449e-bdd7-819177a6f5ca" width="100%" alt="Global Recommendation Routing Matrix">
</div>

---

## Phase 8: Future Works, Transformer Evolution & Ephemeral Architecture

This final section outlines the technological roadmap and scalable system evolutions designed to migrate the current pipeline into a deep learning ecosystem and real-time social platform.

### 8.1 Classifier Evolution (Transitioning to Transformers via RoBERTuito)
* **The Semantic Bottleneck:** While the current production $BoW + NB$ classifier demonstrates exceptional statistical stability, classical n-gram configurations face structural limitations when parsing highly complex contextual dependencies, implicit irony, or nuanced slang.
* **The Transformer Roadmap:** The immediate architectural upgrade consists of deploying a specialized Deep Learning pipeline utilizing **RoBERTuito** (a state-of-the-art, pre-trained BERT variant optimized specifically for informal Spanish and social media text streams). Executing a targeted fine-tuning protocol will capture deep contextual embeddings, maximizing classifier sensitivity and precision.

<div align="center">
  <img src="https://github.com/user-attachments/assets/1765c04e-0c26-489c-94bb-101f9db395a3" width="100%" alt="Transformer Evolution Roadmap">
</div>

---

### 8.2 Dynamic Emotional Segmentation & Ephemeral Spatial Matching
* **Fluid Clustering Framework:** The system intends to transition from static categorical buckets to a dynamic, real-time clustering engine. Users will no longer occupy permanent behavioral groups; instead, they will shift fluidly across reversible clusters determined by their immediate, real-time affective state.
* **The Ephemeral Social Layer:** This computational flexibility opens the door to **Ephemeral Matching**. The recommendation loop will programmatically connect distinct travelers who intersect across three identical vectors: *Space (Geographic location), Time (Temporal availability), and Emotion (Synchronized affective states)*, morphing TRIP(U) into an adaptive, high-response social network.

<div align="center">
  <img src="https://github.com/user-attachments/assets/09700ba6-7661-42dd-b4d2-25f911f02828" width="100%" alt="Dynamic Emotional Clustering & Matching">
</div>

---

### 8.3 Prospective System Architecture Blueprint
* **System Integration:** To support the transition toward deep learning inferences and live multi-criteria recommendations, the prospective system design maps an interconnected data lifecycle.
* **Architectural Flow:** The framework establishes a unified flow integrating raw user text processing via fine-tuned Transformers, live synchronization with the normalized PostgreSQL RDBMS, and a scalable API layer. This ensures that weight updates and multi-criteria rankings function under low-latency constraints during real-world user interaction loops.

<div align="center">
  <img src="https://github.com/user-attachments/assets/c2d8a03f-b787-4740-9c1c-d1915c2c074c" width="100%" alt="Prospective System Architecture Blueprint">
</div>

---

## Conclusions & Key Takeaways

To conclude, **TRIP(U)** stands as a robust proof-of-concept demonstrating three core engineering and architectural realities:

1. **Technical Viability & Domain Generalization:** It proves that an affective classifier trained on heavily specialized textual ecosystems (cinematographic reviews) can successfully generalize and transfer its predictive power to unstructured cross-domain social media environments through strategic entity masking.
2. **True Personalization Integration:** It establishes that dynamic emotional states can be algorithmically normalized and integrated as a heavy-weight structural variable within multi-criteria recommendation loops, shifting traditional static frameworks.
3. **Frictionless Algorithmic Responsibility:** It demonstrates that production-grade ethical guardrails and safety vectors can be seamlessly embedded into runtime logic to protect vulnerable configurations without falling into paternalism or compromising user autonomy.

Ultimately, **TRIP(U)** lays down the architectural foundations for next-generation recommendation systems—systems that move past simple preference optimization to actively comprehend and honor the human affective spectrum.

<div align="center">
  <img src="https://github.com/user-attachments/assets/03b4b81b-fc77-4119-8252-d00ddd257484" width="100%" alt="TRIP_U Project Closure">
</div>

<br>

<div align="center">
  <h3><i>"The variable is you, the journey is your destination."</i></h3>
</div>

---

<div align="center">
  <img src="https://github.com/user-attachments/assets/3d05ad9d-84d4-479e-8019-4ae1b258cf43" width="100%" alt="TRIP_U Contact & Connect">
</div>
