# 💡 Unit 4 Question Bank, Glossary & Viva Voce Exam Vault

> **Course Code:** DS-505 | **Subject:** Big Data Handling and Management for Machine Learning Applications  
> **Course Degree:** B.Sc. (Data Science & Analytics) — Semester V  
> **Institution:** Sutex Bank College of Computer Applications and Science (SBCCAS), Amroli  
> **Affiliation:** Veer Narmad South Gujarat University (VNSGU), Surat  
> **Unit Covered:** Unit 4 — Introduction to Large Language Models and Big Data Applications  
> **Document Purpose:** Comprehensive Exam Preparation, Quick Revision & Practical Viva Voce Guide  
> **Lecture Note Reference:** [Unit-4_Introduction_to_Large_Language_Models_and_Big_Data_Applications.md](../2_Lecture_Notes/Unit-4_Introduction_to_Large_Language_Models_and_Big_Data_Applications.md)  
> **Practical Lab Reference:** [Unit-4_LLM_Text_Processing_and_Pretrained_Models_Lab.md](../3_Projects_Presentations/Unit-4_LLM_Text_Processing_and_Pretrained_Models_Lab.md)  
> **Theory Assignment Reference:** [Assignment-4_Unit-4_Theory_Assignment.md](../4_Assignments/Assignment-4_Unit-4_Theory_Assignment.md)  

---

<details open>
<summary><b>📑 Table of Contents (Click to Expand / Collapse)</b></summary>

- [1. High-Yield Technical Glossary (15 Core Definitions)](#1-high-yield-technical-glossary-15-core-definitions)
- [2. Short Answer Theory Questions (2 to 3 Marks Each)](#2-short-answer-theory-questions-2-to-3-marks-each)
- [3. Long Answer Comprehensive Theory Questions (5 to 7 Marks Each)](#3-long-answer-comprehensive-theory-questions-5-to-7-marks-each)
- [4. Practical Examination Viva Voce Questions & Model Answers](#4-practical-examination-viva-voce-questions--model-answers)

</details>

---

## 1. High-Yield Technical Glossary (15 Core Definitions)

These definitions are frequently asked in 1-mark and 2-mark university questions:

| # | Technical Term | Exact University Examination Definition |
| :-: | :--- | :--- |
| **1** | **Large Language Model (LLM)** | A massive deep learning model based on the Transformer architecture, parameterized by billions of weights, trained on web-scale text corpora to understand and generate human language. |
| **2** | **Natural Language Processing (NLP)** | The subfield of computer science and artificial intelligence concerned with the interactions between computers and human language, particularly computational understanding and synthesis. |
| **3** | **Token** | The atomic building block of text processed by a language model, which can represent a single character, subword morpheme, or whole word. |
| **4** | **Tokenization** | The process of parsing raw unstructured character sequences into a discrete sequence of tokens, which are subsequently mapped to numerical integer indices. |
| **5** | **Text Cleaning** | The systematic preparation and sanitization of raw text data by removing noise, markup, punctuation, irrelevant symbols, and normalizing casing and whitespace. |
| **6** | **Pretrained Model** | A neural network whose weights have already been optimized on a massive general-purpose dataset, capable of direct inference or fine-tuning on downstream tasks. |
| **7** | **Self-Attention Mechanism** | The core algorithmic component of the Transformer that computes dynamic mathematical attention weights between all word pairs in a sequence, capturing long-range contextual relationships. |
| **8** | **Transfer Learning** | A machine learning paradigm where knowledge acquired while solving one task on a vast dataset is transferred and applied to a different but related specialized task. |
| **9** | **Fine-Tuning (SFT)** | The supervised process of taking a general pretrained language model and further training its weights on a smaller, high-quality, domain-specific dataset. |
| **10** | **Abstractive Summarization** | A summarization approach where the model comprehends the source text and generates entirely new sentences and phrases to express the core concepts concisely. |
| **11** | **Extractive Summarization** | A summarization approach that scores existing sentences directly within the source text and extracts the highest-ranking sentences verbatim without altering wording. |
| **12** | **Hugging Face Transformers** | An industry-standard open-source Python library that provides unified APIs, model architectures, and pretrained weights for Transformer models in PyTorch and TensorFlow. |
| **13** | **Pipeline API (`pipeline()`)** | A high-level abstraction in Hugging Face that encapsulates tokenization, model inference, and output post-processing into a single callable object. |
| **14** | **Hallucination** | A phenomenon where an LLM generates grammatically fluent and authoritative-sounding responses that are factually false, ungrounded, or completely fabricated. |
| **15** | **Temperature** | A hyperparameter that scales logits prior to softmax during autoregressive decoding; values closer to $0.0$ enforce deterministic greediness, while higher values ($>0.7$) increase diversity and randomness. |

---

## 2. Short Answer Theory Questions (2 to 3 Marks Each)

#### Q1: What is a Large Language Model (LLM) and how does it differ from a traditional rule-based program?
> **Model Answer:**  
> * **Large Language Model (LLM):** A deep neural network (typically a Transformer) trained on massive text corpora using self-supervised objectives to estimate probability distributions over sequences of words. It generalizes across diverse linguistic contexts.  
> * **Difference from Rule-Based Programs:** Traditional programs rely on explicit, hardcoded `if-else` logical rules written by human programmers, failing when presented with inputs outside predefined patterns. LLMs learn statistical patterns and contextual semantics directly from data, enabling probabilistic understanding of novel, ambiguous, and varied human expressions.

#### Q2: Differentiate between Extractive and Abstractive Summarization with examples.
> **Model Answer:**  
> * **Extractive Summarization:** Selects and extracts important sentences verbatim from the source text based on statistical importance or sentence scoring.  
>   * *Example:* Pulling sentences 1 and 4 directly from a news article without changing any words.  
> * **Abstractive Summarization:** Comprehends the semantic meaning of the document and generates new, paraphrased sentences expressing the central message concisely (similar to human summarization).  
>   * *Example:* Reading a 500-word product review and generating: *"The customer praised the build quality but expressed dissatisfaction with delivery delays."*

#### Q3: What is the functional difference between Python's `split()` and subword tokenization (like BPE)?
> **Model Answer:**  
> * **`split()`:** Divides a string strictly along whitespace or specified delimiters into full words. If a word is unseen during training, it cannot be processed, causing severe **Out-Of-Vocabulary (OOV)** failures.  
> * **Subword Tokenization (Byte-Pair Encoding / BPE):** Breaks complex, rare, or unseen words into frequent subword fragments (e.g., `"unexplainability"` $\rightarrow$ `["un", "##explain", "##ability"]`). This keeps vocabulary sizes compact ($\sim 30,000$ to $50,000$ tokens) while ensuring zero out-of-vocabulary words.

#### Q4: What does the `temperature` parameter control in autoregressive text generation?
> **Model Answer:**  
> * **Temperature ($T$):** Controls the randomness and creativity of next-token probability selection by dividing candidate logits prior to the softmax function:  
>   $$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$  
> * **Low Temperature ($T \rightarrow 0$ or $0.2$):** Sharpened probability distribution; the model greedily picks the most predictable, conservative, and repetitive tokens (ideal for factual Q&A and coding).  
> * **High Temperature ($T \ge 0.7$):** Flattened probability distribution; lower-probability tokens have a higher chance of selection, producing creative, varied, but potentially less coherent text.

#### Q5: Why is `lower()` commonly used in text cleaning pipelines before NLP analysis?
> **Model Answer:**  
> * In standard Python, strings are case-sensitive; `"Data"`, `"DATA"`, and `"data"` are evaluated as three completely distinct byte sequences.  
> * Applying `.lower()` (case folding) standardizes all alphabetic characters into uniform lowercase. This collapses redundant case variants into a single vocabulary key, drastically reducing dictionary sparsity and ensuring accurate term frequency counts.

#### Q6: What is AI Hallucination and what causes it in Large Language Models?
> **Model Answer:**  
> * **Hallucination:** An error where an LLM generates confident, fluent statements that contradict real-world facts or source context.  
> * **Underlying Causes:**  
>   1. **Next-Token Prediction Nature:** LLMs optimize for linguistic probability, not factual truth.  
>   2. **Noisy Training Data:** Falsehoods, contradictions, or outdated facts present in the raw web training data.  
>   3. **Lack of Grounded World Memory:** The model lacks access to real-time external verification unless augmented with tools like RAG (Retrieval-Augmented Generation).

#### Q7: What is the primary role of the Hugging Face `pipeline()` function?
> **Model Answer:**  
> The `pipeline()` function is a unified high-level abstraction that orchestrates the complete end-to-end NLP workflow into a single callable object:  
> 1. **Preprocessing:** Automatically feeds raw text to the model's matching tokenizer.  
> 2. **Model Forward Pass:** Passes token tensors through the deep neural network architecture.  
> 3. **Post-Processing:** Decodes numerical logits into human-readable outputs (e.g., class labels, confidence scores, or generated strings).

---

## 3. Long Answer Comprehensive Theory Questions (5 to 7 Marks Each)

#### Q1: Explain the 5-Stage Working Pipeline and Architecture of Large Language Models with a neat labeled block diagram.
> **Structuring Your Answer for Full Marks:**
> 1. **Introduction:** Define an LLM and the transition from statistical language models (N-grams) to modern deep neural networks.
> 2. **Block Diagram:** Draw the complete 5-stage pipeline:
>    $$\text{User Prompt} \longrightarrow \text{Tokenization} \longrightarrow \text{Vector Embeddings} \longrightarrow \text{Transformer Processing} \longrightarrow \text{Next-Token Generation}$$
> 3. **Stage-by-Stage Explanation:**
>    * **Stage 1 (Prompt Ingestion):** Captures raw natural language user instruction.
>    * **Stage 2 (Tokenization):** Subword decomposition (BPE/WordPiece) converting strings into token ID sequences.
>    * **Stage 3 (Embedding & Positional Encoding):** Projecting discrete IDs into continuous dense vector spaces and adding positional vectors to preserve word order.
>    * **Stage 4 (Transformer Blocks & Self-Attention):** Deep stacked Transformer decoder layers utilizing Multi-Head Self-Attention ($Q, K, V$) and Feed-Forward Networks to update token representations.
>    * **Stage 5 (Softmax & Output Sampling):** Linear projection to vocabulary size, computing softmax probabilities, and applying decoding strategies (greedy, top-$k$, top-$p$).
> 4. **Iterative Loop:** Explain that generation is **autoregressive**—each newly generated token is appended to the input context to generate the next token.

#### Q2: Discuss the Relationship between Big Data and Large Language Models. Why are massive datasets essential for LLM pretraining?
> **Structuring Your Answer for Full Marks:**
> 1. **The Core Relationship:** Explain that Big Data acts as the foundational fuel and empirical teacher for LLMs. Without terabyte/petabyte-scale textual data, modern LLMs cannot generalize.
> 2. **Why Massive Datasets are Essential:**
>    * **Learning World Knowledge:** Billions of parameters require hundreds of billions of training tokens to prevent catastrophic overfitting (scaling laws).
>    * **Syntactic and Semantic Mastery:** Exposure to diverse dialects, formal logic, scientific literature, code, and informal conversational text.
>    * **Implicit Reasoning:** Emergent abilities (e.g., in-context learning, few-shot prompting) only manifest when models exceed threshold parameter counts trained on massive data.
> 3. **Data Preprocessing Bottlenecks (The Big Data Engineering Problem):**
>    * Crawling and ingesting Common Crawl, Wikipedia, books, and code.
>    * Distributed deduplication using MinHash / LSH on clusters (Apache Spark).
>    * Filtering toxic content, personally identifiable information (PII), and machine-generated gibberish.
> 4. **Summary Table:** Contrast model parameters vs. required training dataset tokens across landmark models (GPT-2, GPT-3, LLaMA).

#### Q3: Compare and contrast Rule-Based Chatbots versus Modern LLM-Powered Conversational Agents.
> **Structuring Your Answer for Full Marks:**
> 1. **Introduction:** Contrast deterministic rule-matching engines with probabilistic generative models.
> 2. **Architectural Comparison Table:**
>
> | Dimension | Rule-Based Chatbots | LLM-Powered Chatbots |
> | :--- | :--- | :--- |
> | **Core Technology** | Hardcoded decision trees, regular expressions, slot filling | Deep Transformer neural networks, self-attention |
> | **Input Flexibility** | Rigid; fails if user phrasing varies slightly from syntax | Extremely flexible; handles typos, colloquialisms, idioms |
> | **Context Window** | Very limited; single-turn or fixed state machine slots | Multi-turn conversational history retained in context buffer |
> | **Training Data** | Pre-written manual rule scripts | Terabytes of diverse conversational and informational text |
> | **Hallucination Risk** | Zero; answers are strictly authored by humans | High; model may generate inaccurate but plausible text |
> | **Development Cost** | Low compute setup; very high ongoing manual rule authoring | High initial training/API cost; low per-intent authoring |
>
> 3. **Hybrid Enterprise Architecture:** Explain why modern banking and enterprise systems combine rule-based deterministic guardrails with LLM natural language understanding.

#### Q4: Discuss Ethical Considerations, Algorithmic Bias, and Safety Challenges in Large Language Models and AI Systems.
> **Structuring Your Answer for Full Marks:**
> 1. **Introduction:** Define AI ethics as the set of moral principles and systemic guardrails ensuring AI systems operate transparently, fairly, and beneficially.
> 2. **Major Ethical Challenges:**
>    * **Algorithmic Bias & Representation Disparity:** Training datasets scraped from the internet encode historical gender, racial, and socio-economic prejudices, causing biased hiring or credit filtering.
>    * **Misinformation & Hallucinations:** High-speed dissemination of convincing but false technical, medical, or political claims.
>    * **Copyright & Intellectual Property:** Scraping copyrighted creative works and proprietary source code without creator attribution or compensation.
>    * **Privacy Violations & Data Leakage:** Memorization of private personal data (PII, phone numbers, addresses) leaked during generation.
>    * **Environmental Impact:** Massive carbon footprint and electricity/water consumption during month-long GPU cluster pretraining.
> 3. **Technical Mitigation Strategies:** RLHF (Reinforcement Learning from Human Feedback), red-teaming, input prompt filters, and differential privacy.

#### Q5: Detail the Core Python String Functions (`split`, `lower`, `replace`, `count`) and explain how they form an integrated text preprocessing pipeline for Big Data.
> **Structuring Your Answer for Full Marks:**
> 1. **Syntax & Mechanics of Each Function:**
>    * `lower()`: Method signature, string immutability, case folding.
>    * `replace(old, new)`: Exact substring matching, noise and punctuation stripping.
>    * `split(sep)`: Splitting string into list of tokens, whitespace default behavior.
>    * `count(sub)`: Frequency tallying of target substring within a larger text.
> 2. **Code Demonstration:** Provide a clean, end-to-end Python script that accepts a noisy multi-line string, cleans it, tokenizes it, and counts term occurrences.
> 3. **Relevance in Big Data Ingestion:** Explain that in Big Data frameworks (like PySpark or MapReduce), these string primitives are vectorized inside `map()` or DataFrame User-Defined Functions (UDFs) to clean millions of raw text lines before feeding tokenizers.

---

## 4. Practical Examination Viva Voce Questions & Model Answers

These questions are specifically tailored for external practical examiners during laboratory evaluations:

#### ❓ Q1: What is the return type of Python's string `.split()` function?
> **Answer:** It returns a Python **`list`** of strings. If no delimiter is specified, it splits on arbitrary whitespace (including spaces, tabs, and newlines).

#### ❓ Q2: If `text = "Data Science"`, what is the output of `text.replace("Science", "Analytics")`?
> **Answer:** `"Data Analytics"`.

#### ❓ Q3: Does `string.lower()` modify the original string in place?
> **Answer:** **No.** In Python, strings are **immutable**. Calling `.lower()` returns a new string object containing the lowercase characters; the original variable remains unchanged unless reassigned.

#### ❓ Q4: How do you determine the total number of words in a space-separated string `s` using basic Python?
> **Answer:** `len(s.split())`.

#### ❓ Q5: What command installs the Hugging Face transformers package in a Jupyter notebook or Google Colab?
> **Answer:** `!pip install transformers` (often accompanied by `torch`: `!pip install transformers torch`).

#### ❓ Q6: What is the primary role of the Hugging Face `pipeline` function?
> **Answer:** It abstracts the entire NLP workflow—automatically linking the appropriate tokenizer, model architecture, model weights, inference execution, and output decoding into a single callable interface.

#### ❓ Q7: Why do modern LLMs use subword tokenization (like BPE) instead of traditional whitespace `split()`?
> **Answer:** Whitespace `split()` creates an unmanageably huge vocabulary and fails on unknown/misspelled words with Out-of-Vocabulary (OOV) tokens. Subword tokenization breaks rare words into known subword morphemes, ensuring complete coverage with a compact, fixed-size vocabulary.

#### ❓ Q8: What is the difference between greedy decoding and nucleus sampling in text generation?
> **Answer:** Greedy decoding always selects the single token with the highest probability ($P_{\max}$), which is deterministic but often leads to repetitive loops. Nucleus (top-$p$) sampling dynamically samples from the smallest set of top tokens whose cumulative probability exceeds threshold $p$ (e.g., $p=0.9$), producing natural diversity.

#### ❓ Q9: What happens if you set `max_length` smaller than `min_length` in a summarization pipeline?
> **Answer:** The library throws a `ValueError` during configuration because the maximum allowable generated tokens cannot be fewer than the required minimum tokens.

#### ❓ Q10: What does `do_sample=False` signify in a Hugging Face text generation call?
> **Answer:** It turns off probabilistic sampling and activates deterministic **greedy decoding** (or beam search if `num_beams > 1`), ensuring identical output across repeated runs with the same prompt.

#### ❓ Q11: What is the purpose of `pad_token_id=50256` when calling GPT-2 in Hugging Face?
> **Answer:** GPT-2 does not natively have an explicit padding token configured. Setting `pad_token_id` to the End-Of-Sequence token ID (`50256`) prevents terminal warning messages when batches of variable lengths are generated.

#### ❓ Q12: How does `count()` differ from finding term frequency using a dictionary or `collections.Counter`?
> **Answer:** `string.count("sub")` searches for literal substring matches across the raw text (which can produce false positives inside other words, e.g., `"in"` inside `"learning"`). In contrast, tokenizing with `.split()` and using `Counter(tokens)` counts whole, distinct token occurrences.

#### ❓ Q13: Can Hugging Face pipelines run on CPU without an expensive GPU?
> **Answer:** **Yes.** By default, `pipeline()` runs inference on the host CPU. If a CUDA-compatible GPU is available, passing parameter `device=0` accelerates inference significantly.

#### ❓ Q14: What is the difference between a Pretrained Model and a Fine-Tuned Model?
> **Answer:** A **Pretrained Model** has been trained on broad web-scale corpora for general language understanding (e.g., base GPT-2 or BERT). A **Fine-Tuned Model** has had its weights further specialized and adapted using supervised labeled data for a specific downstream application (e.g., clinical diagnosis or legal summarization).

#### ❓ Q15: Why is prompt engineering important when working with LLM APIs?
> **Answer:** Because LLMs are autoregressive probability models, the precision, structure, contextual constraints, and phrasing of the input prompt directly condition the probability distribution of the subsequent generated tokens.

---

<div align="center">

Made with 💙 for the **B.Sc. Data Science & Analytics** Students  
**Sutex Bank College of Computer Applications and Science (SBCCAS)**  
*Veer Narmad South Gujarat University (VNSGU), Surat*

</div>
