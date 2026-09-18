<div align="center">

# SUTEX BANK COLLEGE OF COMPUTER APPLICATIONS & SCIENCE, AMROLI
### T.Y.B.Sc. (DATA SCIENCE & ANALYTICS) — SEMESTER: 5th
## ASSIGNMENT – IV (UNIT – IV)
### SUBJECT: DS-505 — Big Data Handling and Management for Machine Learning Applications

**Unit Covered:** Unit 4 – Introduction to Large Language Models and Big Data Applications  
**Submission Date:** *To be announced by Course Instructor*  
**Recommended Student Notebook Length:** **12 to 15 Pages Total**  

---

</div>

> **Instructions for Students:**
> 1. Handwrite all answers neatly in your official assignment notebook / sheets.
> 2. Each **Long Question** should take **~2 to 2.5 pages** (including diagrams and tables).
> 3. Each **Short Question** should take **~0.5 to 0.75 page**.
> 4. Total assignment volume should be between **12 to 15 handwritten pages**.
> 5. All questions are directly searchable and referenced from your lecture notes:  
>    👉 [**`2_Lecture_Notes/Unit-4_Introduction_to_Large_Language_Models_and_Big_Data_Applications.md`**](../2_Lecture_Notes/Unit-4_Introduction_to_Large_Language_Models_and_Big_Data_Applications.md)

---

## SECTION I: Answer in Detail (Long Questions)
*(Attempt all questions. Expected length: ~2 to 2.5 handwritten pages per question)*

### 1. Explain the Concept, Architecture, and 5-Step Working Mechanism of Large Language Models (LLMs).
* Define a Language Model (LM) and a Large Language Model (LLM).
* Explain the 5-step working pipeline:
  1. Input Text / Prompt
  2. Tokenization
  3. Numerical Representation (Vectors & Embeddings)
  4. Transformer Processing (Self-Attention mechanism)
  5. Output Generation (Probability distribution and next-token prediction)
* Draw the neat architectural block diagram illustrating this 5-step flow.
* Briefly explain why the Self-Attention mechanism is crucial in Transformers.  
*(📖 **Reference:** Lecture Notes `Section 4.1.1` & `Figure 1`)*

---

### 2. Discuss LLM Real-World Applications with a focus on Conversational Agents and Text Summarization.
* Outline major practical applications of LLMs in industry.
* Detail **Text Summarization**:
  * Define and differentiate **Extractive Summarization** vs. **Abstractive Summarization** with suitable examples.
* Detail **Conversational Agents (Chatbots)**:
  * Compare traditional **Rule-Based Chatbots** with modern **LLM-Powered Chatbots** using a comparison table (covering architecture, flexibility, training data, and context handling).  
*(📖 **Reference:** Lecture Notes `Section 4.1.2`)*

---

### 3. Explain the Relationship between Big Data and Large Language Models.
* Why are massive Big Data corpora an essential prerequisite for training modern LLMs?
* Explain the concepts of:
  * **Self-Supervised Pre-training** on web-scale raw datasets.
  * **Fine-Tuning (SFT)** on specialized domain data.
* Discuss three major engineering challenges when collecting, cleaning, and managing terabyte-scale text datasets.  
*(📖 **Reference:** Lecture Notes `Section 4.1.3` & `Section 4.2.1`)*

---

### 4. Explain Text Preprocessing, Cleaning, and Tokenization Techniques in Modern NLP.
* Why does raw text require cleaning before being fed into machine learning / deep learning models?
* Describe standard text cleaning steps (lowercasing, punctuation stripping, markup removal).
* Compare the three main tokenization approaches:
  1. Word-Level Tokenization
  2. Character-Level Tokenization
  3. Subword Tokenization (e.g., Byte-Pair Encoding / WordPiece)
* Draw a comparison table analyzing vocabulary size, Out-Of-Vocabulary (OOV) handling, and sequence length for these three tokenization methods.  
*(📖 **Reference:** Lecture Notes `Section 4.2.2` & `Section 4.4.2`)*

---

### 5. Write a Detailed Note on the Hugging Face Ecosystem, Pretrained APIs, and Ethical Considerations in AI.
* What is Hugging Face and what is the role of the `transformers` library?
* Explain the high-level `pipeline()` function and write a short Python snippet showing how to use `pipeline("text-generation")` or `pipeline("summarization")`.
* Compare **API-Based Model Consumption** vs. **Local Model Hosting** using a table.
* Discuss key **Ethical Considerations in AI**:
  * Algorithmic Bias (with a real-world scenario)
  * AI Hallucinations (meaning and risks in healthcare/finance)
  * Data Privacy & Copyright  
*(📖 **Reference:** Lecture Notes `Section 4.3.1`, `4.3.2`, and `4.3.3`)*

---

## SECTION II: Answer Briefly (Short Questions)
*(Attempt all questions. Expected length: ~0.5 to 0.75 handwritten page per question)*

### 1. Explain the three components in the name "Large Language Model" (Large, Language, and Model).
* Briefly explain what each word signifies using a structured 3-row table.  
*(📖 **Reference:** Lecture Notes `Section 4.1.1`, Table 0)*

---

### 2. Differentiate between Python's built-in `split()` method and NLP Subword Tokenization.
* Explain why simple whitespace splitting fails on Out-of-Vocabulary (OOV) and punctuation-attached words.  
*(📖 **Reference:** Lecture Notes `Section 4.2.2` & `Section 5.4`)*

---

### 3. What do Temperature and Nucleus Sampling (Top-p) control in text generation APIs?
* Explain the difference between setting temperature close to `0.0` vs. close to `1.0`.
* Explain how nucleus sampling dynamically selects candidate tokens.  
*(📖 **Reference:** Lecture Notes `Section 4.3.2` & `Section 5.4`)*

---

### 4. What is AI Hallucination and Algorithmic Bias in Large Language Models?
* Define hallucination and explain why it is dangerous in critical decision-making fields.
* Define algorithmic bias and state how it originates from web training data.  
*(📖 **Reference:** Lecture Notes `Section 4.3.3` & `Section 5.1`)*

---

### 5. Explain the role of Python string methods `lower()`, `replace()`, and `count()` in text cleaning.
* Give a brief one-line explanation and code example for each function.  
*(📖 **Reference:** Lecture Notes `Section 4.4.2` & `Section 5.4`)*

---

<br>

<div align="right">

**Asst. Prof. Hitesh Patel**  
*Department of Computer Science & Data Science*  
*Sutex Bank College of Computer Applications & Science (SBCCAS), Amroli*  
*F.Y. / T.Y. B.Sc. (Data Science & Analytics) & B.C.A.*  

</div>
