# 📝 Theory Assignment 4: Introduction to Large Language Models and Big Data Applications

> **Course Code:** DS-505  
> **Course Title:** Big Data Handling and Management for Machine Learning Applications  
> **Degree Program:** B.Sc. (Data Science & Analytics) — Semester V  
> **Institution:** Sutex Bank College of Computer Applications and Science (SBCCAS)  
> **Affiliation:** Veer Narmad South Gujarat University (VNSGU), Surat  
> **Academic Year:** 2026 – 2027  
> **Unit Covered:** **Unit 4 – Introduction to Large Language Models and Big Data Applications**  
> **Total Marks:** 50 Marks *(Weightage: External/Internal Continuous Assessment)*  
> **Recommended Notebook Volume:** **12 to 15 Handwritten Pages**  

---

## 📌 Instructions for Students

1. **Submission Format:** Handwrite the assignment legibly in your official course notebook/assignment sheets.
2. **Page Budgeting & Limit:** 
   - Each **Long Question** should take approximately **2 to 2.5 handwritten pages** (including diagrams and tables).
   - Each **Short Question** should take approximately **0.5 to 0.75 handwritten page**.
   - The total assignment must be completed within **12 to 15 notebook pages**.
3. **Reference Material:** All answers are directly available and easily searchable in the provided course lecture notes:
   👉 **[`2_Lecture_Notes/Unit-4_Introduction_to_Large_Language_Models_and_Big_Data_Applications.md`](../2_Lecture_Notes/Unit-4_Introduction_to_Large_Language_Models_and_Big_Data_Applications.md)**
4. **Diagrams & Tables:** Wherever indicated, draw neat, labeled diagrams and construct comparative tables to secure maximum marks.
5. **Academic Integrity:** Write the answers in your own words after studying the material. Direct copy-pasting from online AI tools without conceptual understanding is discouraged; viva questions will be asked based on your submissions.

---

## 🧭 Search & Reference Guide (Mapping to Lecture Notes)

Students can quickly locate the required concepts in the Unit 4 lecture notes using the section references below:

| Question # | Type | Topic | Exact Section in Lecture Notes | Search Keyword |
| :--- | :---: | :--- | :--- | :--- |
| **LQ 1** | Long | LLM Definition & 5-Step Working Mechanism | `Section 4.1.1` & `Figure 1` | `"How Does an LLM Work?"` |
| **LQ 2** | Long | Chatbots & Text Summarization (Extractive vs Abstractive) | `Section 4.1.2` | `"Examples of LLM applications"` |
| **LQ 3** | Long | Big Data Scale & Pretraining vs Fine-Tuning | `Section 4.1.3` & `Section 4.2.1` | `"Why Big Data is Important for LLMs"` |
| **LQ 4** | Long | Text Preprocessing, Cleaning & Tokenization Techniques | `Section 4.2.2` & `Section 4.4.2` | `"Tokenization and Text Cleaning"` |
| **LQ 5** | Long | Hugging Face Ecosystem, Pretrained APIs & AI Ethics | `Section 4.3.1`, `4.3.2`, `4.3.3` | `"Introduction to Hugging Face"` |
| **SQ 1** | Short | Deconstruction of the Term "Large Language Model" | `Section 4.1.1` (Table 0) | `"Why is it called the Large Language Model"` |
| **SQ 2** | Short | Python `split()` vs Subword Tokenization | `Section 4.2.2` & `Section 5.4` | `"subword tokenization"` / `OOV` |
| **SQ 3** | Short | Decoding Parameters: Temperature & Nucleus Sampling | `Section 4.3.2` & `Section 5.4` | `"temperature"` / `"nucleus sampling"` |
| **SQ 4** | Short | AI Hallucinations and Algorithmic Bias | `Section 4.3.3` & `Section 5.1` | `"Hallucination"` / `"Ethical Considerations"` |
| **SQ 5** | Short | String Primitives: `lower()`, `replace()`, and `count()` | `Section 4.4.2` | `"Text Processing using Python"` |

---

## 📑 SECTION A: Long Answer Questions (5 × 7 = 35 Marks)

*Answer all 5 questions. Each question carries 7 marks. Recommended length per question: ~2 to 2.5 notebook pages.*

---

### ❓ Question 1: Concept, Working Mechanism & Architecture of Large Language Models
*(7 Marks | Recommended Length: ~2 to 2.5 Pages)*

1. **Definition:** Define a **Language Model (LM)** and a **Large Language Model (LLM)** from both a probabilistic and machine learning perspective.
2. **The 5-Step Working Mechanism:** Detail the step-by-step pipeline an LLM executes when responding to an input prompt:
   - **Step 1:** Input Text / Prompt Reception
   - **Step 2:** Tokenization (atomic text division)
   - **Step 3:** Numerical Representation (Vectors & Word/Token Embeddings)
   - **Step 4:** Transformer Processing & Multi-Head Self-Attention
   - **Step 5:** Output Generation (Probability distribution over vocabulary & next-token prediction)
3. **Visual Representation:** Draw the neat, labeled architectural flowchart/block diagram illustrating these 5 steps.
4. **Attention Mechanism:** Briefly explain why the **Self-Attention** mechanism in the Transformer architecture outperforms legacy sequence models (like standard RNNs).

> 🔍 **Reference:** Check `Section 4.1.1` *(Concept of Large Language Models)* and `Figure 1` in the lecture notes.

---

### ❓ Question 2: LLM Real-World Applications: Conversational Agents and Text Summarization
*(7 Marks | Recommended Length: ~2 to 2.5 Pages)*

1. **Major Application Categories:** Outline the key domains where LLMs are deployed in data science (Chatbots, Summarization, Question-Answering, Code Synthesis, Classification).
2. **Text Summarization In-Depth:**
   - Define text summarization and explain the crucial operational difference between **Extractive Summarization** and **Abstractive Summarization**.
   - Provide an illustrative example demonstrating how both approaches process the same source text paragraph.
3. **Conversational Agents (Chatbots):**
   - Explain how traditional **Rule-Based Chatbots** differ from modern **LLM-Powered Conversational Agents**.
   - Construct a structured comparative table covering: *Architecture, Flexibility, Context Handling, Training Data Requirements, and Failure Modes*.

> 🔍 **Reference:** Check `Section 4.1.2` *(Examples of LLM applications)* in the lecture notes.

---

### ❓ Question 3: The Intersection of Big Data and LLMs: Massive Corpora, Pre-Training, and Scaling
*(7 Marks | Recommended Length: ~2 Pages)*

1. **The Role of Big Data:** Why is Big Data an absolute prerequisite for training Large Language Models? Explain how dataset scale correlates with model intelligence and emergent abilities.
2. **Training Corpora:** What types of datasets constitute pretraining corpora (web scrapes, books, encyclopedias, domain-specific text, code repositories)?
3. **Pre-Training vs. Fine-Tuning:**
   - Define **Self-Supervised Pre-training** on raw Big Data text.
   - Define **Supervised Fine-Tuning (SFT)** and **Instruction Tuning** for specialized tasks.
   - Explain why pretraining requires distributed compute clusters (like Spark/GPUs) while fine-tuning can be conducted on smaller resources.
4. **Engineering Bottlenecks:** Discuss at least three practical challenges encountered when processing multi-terabyte text datasets (memory limits, duplicate noise, text normalization).

> 🔍 **Reference:** Check `Section 4.1.3` *(Importance of Large Datasets for LLM Training)* and `Section 4.2.1` *(Working with Large Text Datasets)* in the lecture notes.

---

### ❓ Question 4: Text Data Cleaning, Normalization & Tokenization Strategies
*(7 Marks | Recommended Length: ~2 Pages)*

1. **Need for Text Cleaning:** Why must raw, unstructured text undergo rigorous preprocessing before being input into deep learning models?
2. **Standard Cleaning Pipeline:** Explain the essential cleaning steps:
   - Case normalization (lowercasing)
   - Punctuation and special character removal
   - Stripping HTML/XML markup tags and escape characters
   - Handling numbers and non-printable whitespace characters
3. **Tokenization Paradigms:** Compare the three primary tokenization approaches:
   - **Word-Level Tokenization**
   - **Character-Level Tokenization**
   - **Subword Tokenization** (Byte-Pair Encoding / WordPiece)
4. **Pros & Cons Analysis:** Draw a comparison table analyzing vocabulary size, out-of-vocabulary (OOV) handling, and sequence length across these three tokenization methods.

> 🔍 **Reference:** Check `Section 4.2.2` *(Tokenization and Text Cleaning)* and `Section 4.4.2` in the lecture notes.

---

### ❓ Question 5: Hugging Face Ecosystem, Pretrained APIs & Ethical AI Considerations
*(7 Marks | Recommended Length: ~2 to 2.5 Pages)*

1. **The Hugging Face Framework:**
   - What is Hugging Face and what role does the `transformers` library play in democratizing AI models?
   - Explain the concept of the high-level `pipeline()` function. Write a brief Python code snippet showing how to initialize and run a `text-generation` or `summarization` pipeline.
2. **API vs. Local Deployment:** Construct a comparative table highlighting the differences between **API-Based Model Consumption** (Cloud Inference) and **Local Model Hosting** across latency, cost, hardware requirements, and data privacy.
3. **Ethical Considerations in Modern AI Systems:**
   - **Algorithmic Bias & Fairness:** How do biases in web training datasets propagate into model outputs? Give a realistic scenario.
   - **Hallucination:** What causes LLMs to fabricate factual inaccuracies with high confidence?
   - **Data Privacy & Governance:** Discuss intellectual property and sensitive user data risks.

> 🔍 **Reference:** Check `Section 4.3.1`, `Section 4.3.2` *(Table 2)*, and `Section 4.3.3` *(Ethical Considerations)* in the lecture notes.

---

## 📑 SECTION B: Short Answer Questions (5 × 3 = 15 Marks)

*Answer all 5 questions. Each question carries 3 marks. Recommended length per question: ~0.5 to 0.75 notebook page.*

---

### ❓ Question 6: Deconstruction of the Term "Large Language Model"
*(3 Marks | Recommended Length: ~0.5 Page)*

Explain the significance of each of the three words in **Large Language Model (LLM)**:
- **Large:** What parameters and data dimensions make it "large"?
- **Language:** What linguistic scope does it handle?
- **Model:** What type of computational/mathematical entity is it?
*(Present your answer using the concise 3-row summary table provided in the notes).*

> 🔍 **Reference:** Check `Section 4.1.1` *(Table: Word vs Meaning)* in the lecture notes.

---

### ❓ Question 7: Python `split()` vs. Subword Tokenization
*(3 Marks | Recommended Length: ~0.5 to 0.75 Page)*

1. What is the fundamental functional difference between Python's built-in `string.split()` method and an NLP tokenizer like Byte-Pair Encoding (BPE)?
2. Why does simple whitespace splitting fail when encountering Out-of-Vocabulary (OOV) words or punctuation-joined words (e.g., `"data-science,"`)?

> 🔍 **Reference:** Check `Section 4.2.2` and `Section 5.4` *(Viva Voce)* in the lecture notes.

---

### ❓ Question 8: Generation Decoding Parameters: Temperature and Nucleus Sampling
*(3 Marks | Recommended Length: ~0.5 to 0.75 Page)*

1. What is the **temperature** parameter in text generation APIs? Explain the difference in output behavior when temperature is set near `0.0` (deterministic/greedy) versus near `1.0` (creative/random).
2. What is **Nucleus Sampling (Top-p)**, and how does it dynamically prevent low-probability nonsensical tokens from being generated?

> 🔍 **Reference:** Check `Section 4.3.2` and `Section 5.4` in the lecture notes.

---

### ❓ Question 9: AI Hallucinations and Algorithmic Bias
*(3 Marks | Recommended Length: ~0.5 Page)*

1. Define the term **AI Hallucination** in the context of Large Language Models.
2. Why are hallucinations especially hazardous in healthcare and legal domains?
3. Briefly define **Algorithmic Bias** and cite one common cause of bias during the data collection phase.

> 🔍 **Reference:** Check `Section 4.3.3` and `Section 5.1` *(Glossary Table)* in the lecture notes.

---

### ❓ Question 10: Essential Python String Methods for Text Preprocessing
*(3 Marks | Recommended Length: ~0.5 Page)*

Explain the purpose, behavior, and output of the following three built-in Python string methods when building a text cleaning pipeline:
1. `str.lower()` — Does it modify the original string in place? Explain why.
2. `str.replace(old, new)` — Write a 1-line example cleaning a punctuation mark from a sentence.
3. `str.count(sub)` — How is it utilized to analyze word or token frequency in a document?

> 🔍 **Reference:** Check `Section 4.4.2` and `Section 5.4` *(Viva Voce)* in the lecture notes.

---

## 📊 Evaluation Rubric & Marking Scheme

| Question Category | Criteria | Marks Allocation |
| :--- | :--- | :---: |
| **Long Questions (Q1 to Q5)** | • Clear definitions and conceptual depth<br>• Inclusion of diagrams, equations, or flowcharts<br>• Structured comparison tables where requested<br>• Real-world examples and code snippets | **7 Marks each**<br>*(Total: 35 Marks)* |
| **Short Questions (Q6 to Q10)** | • Precise, to-the-point technical explanations<br>• Correct terminology and syntax demonstration<br>• Neat bullet points or short tables | **3 Marks each**<br>*(Total: 15 Marks)* |
| **Total Marks** | | **50 Marks** |

---

<div align="center">

**Department of Computer Science & Data Science**  
**Sutex Bank College of Computer Applications and Science (SBCCAS)**  
*Veer Narmad South Gujarat University (VNSGU), Surat*

</div>
