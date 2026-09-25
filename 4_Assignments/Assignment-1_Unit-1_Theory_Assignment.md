<div align="center">

# SUTEX BANK COLLEGE OF COMPUTER APPLICATIONS & SCIENCE, AMROLI
### T.Y.B.Sc. (DATA SCIENCE & ANALYTICS) — SEMESTER: 5th
## ASSIGNMENT – I (UNIT – I)
### SUBJECT: DS-505 — Big Data Handling and Management for Machine Learning Applications

**Unit Covered:** Unit 1 – Introduction to Big Data and Large Dataset Handling  
**Submission Date:** *To be announced by Course Instructor*  
**Recommended Student Notebook Length:** **12 to 15 Pages Total**  

---

</div>

> **Instructions for Students:**
> 1. Handwrite all answers neatly in your official course assignment notebook / sheets.
> 2. Each **Long Question** should take **~2 to 2.5 pages** (including diagrams and comparison tables).
> 3. Each **Short Question** should take **~0.5 to 0.75 page**.
> 4. For **Multiple Choice Questions (MCQs)**, write the question, correct option letter, and option text.
> 5. Total assignment volume should be between **12 to 15 handwritten pages**.
> 6. All questions are directly searchable and referenced from your lecture notes:  
>    👉 [**`2_Lecture_Notes/Unit-1_Introduction_to_Big_Data_and_Large_Dataset_Handling.md`**](../2_Lecture_Notes/Unit-1_Introduction_to_Big_Data_and_Large_Dataset_Handling.md)

---

## SECTION I: Answer in Detail (Long Questions)
*(Attempt all questions. Expected length: ~2 to 2.5 handwritten pages per question)*

### 1. Explain the 5 Vs of Big Data in Detail with Real-World Industrial Examples.
* Define Big Data using formal definitions (Gartner / McKinsey / DECAP456).
* Detail each of the 5 V characteristics:
  1. **Volume:** Data scale, magnitude, drivers, and storage challenges.
  2. **Velocity:** Batch vs. real-time streaming paradigms.
  3. **Variety:** Structured, Semi-Structured, and Unstructured data with examples.
  4. **Veracity:** Data cleanliness, noise, and the GIGO principle.
  5. **Value:** Converting raw data into predictive machine learning insights.
* Draw the neat, labeled 5Vs architecture diagram.  
*(📖 **Reference:** Lecture Notes `Section 1.1` & `Section 1.2`)*

---

### 2. Differentiate Traditional Relational Databases (RDBMS) vs. Modern Big Data Systems.
* Why do traditional single-machine RDBMS systems fail when handling Big Data workloads?
* Compare RDBMS vs. Big Data across:
  * Scaling approach (**Vertical / Scale-Up** vs. **Horizontal / Scale-Out**)
  * Schema strategy (**Schema-on-Write** vs. **Schema-on-Read**)
  * Data variety and structure flexibility
  * Read/Write transaction bottlenecks (ACID locking vs. parallel distributed execution)
* Construct a structured comparative table summarizing all dimensions.  
*(📖 **Reference:** Lecture Notes `Section 1.5`)*

---

### 3. Explain Python Memory Architecture and the Out-of-Memory (OOM) Problem in Large Dataset Ingestion.
* Explain the concept of **Memory Amplification** (Why a 1 GB CSV file on disk requires 3 GB to 5 GB of RAM).
* Discuss the internal causes:
  1. Python `PyObject` structure overhead
  2. Default 64-bit numerical allocations (`int64`, `float64`)
  3. Object dtype pointer fragmentation for text strings
* What are the consequences of memory overflow on an operating system (paging, freezing, crash)?  
*(📖 **Reference:** Lecture Notes `Section 2.3`)*

---

### 4. Detail In-Memory Optimization Techniques in Pandas (Downcasting and Categoricals).
* Explain how a data scientist can reduce RAM consumption by 70% to 90% without losing information:
  * **Numeric Downcasting:** Downcasting integers (`int64` &rarr; `int16`/`int32`) and floats (`float64` &rarr; `float32`).
  * **Categorical Dtype:** How dictionary encoding optimizes repetitive text columns.
  * **Selective Loading:** Using the `usecols` parameter in `pd.read_csv()`.
* Provide a clean Python code example demonstrating memory profiling before and after optimization.  
*(📖 **Reference:** Lecture Notes `Section 2.4`)*

---

### 5. Explain the Architecture and Working of Chunked Data Processing using Pandas.
* What is the role of the `chunksize` parameter in `pd.read_csv()`?
* Draw a sequence diagram illustrating:
  `Disk File ➔ Stream Chunk into RAM ➔ Compute Local Aggregates ➔ Free RAM ➔ Final Accumulator`
* Write a complete Python script demonstrating how to calculate **total revenue** and **record counts** across a large file using a chunked loop.  
*(📖 **Reference:** Lecture Notes `Section 2.5`)*

---

## SECTION II: Answer Briefly (Short Questions)
*(Attempt all questions. Expected length: ~0.5 to 0.75 handwritten page per question)*

### 1. Differentiate between Structured, Semi-Structured, and Unstructured Data.
* Provide the defining characteristic, internal format, and one real-world example for each.  
*(📖 **Reference:** Lecture Notes `Section 1.2`)*

---

### 2. Compare Apache Parquet and Plain Text CSV formats for Big Data storage.
* Contrast both formats based on storage type (Row-based vs. Columnar), compression ratio, and read performance.  
*(📖 **Reference:** Lecture Notes `Section 2.2`)*

---

### 3. What is the fundamental difference between NumPy Vectorized Computing and Standard Python Lists?
* Explain how contiguous memory blocks and CPU SIMD instructions enable NumPy to execute 25x–80x faster than pure Python loops.  
*(📖 **Reference:** Lecture Notes `Section 3.1`)*

---

### 4. Differentiate between `df.dropna(how='any')` and `df.dropna(how='all')`.
* Explain the filtering condition for both parameters with a mini 2-row table example.  
*(📖 **Reference:** Lecture Notes `Section 4.3` & `Section 6.2`)*

---

### 5. Why is the `lines=True` argument critical when reading JSON Lines (`.jsonl`) files in Big Data?
* Explain the memory difference between loading a single standard JSON array `[...]` vs. streaming line-delimited JSON records.  
*(📖 **Reference:** Lecture Notes `Section 4.1`)*

---

## SECTION III: Multiple Choice Questions (MCQs)
*(Choose the single correct option. Write the question number, option letter, and answer text).*

#### 1. Which of the 5 Vs of Big Data addresses data quality, noise, and trustworthiness?
- **A)** Velocity
- **B)** Veracity
- **C)** Variety
- **D)** Volume

#### 2. What is the memory size allocated per numeric value by default in Pandas for `float64`?
- **A)** 2 bytes
- **B)** 4 bytes
- **C)** 8 bytes
- **D)** 16 bytes

#### 3. In Pandas, converting a high-cardinality repetitive string column from `object` to `category` achieves memory reduction through:
- **A)** Lossy data compression
- **B)** Dictionary-based integer encoding
- **C)** Truncating long strings
- **D)** Floating-point normalization

#### 4. When reading a 10 GB file on an 8 GB RAM laptop using `pd.read_csv()`, which parameter prevents an Out-Of-Memory crash?
- **A)** `low_memory=False`
- **B)** `chunksize`
- **C)** `index_col=0`
- **D)** `engine='c'`

#### 5. Which of the following storage file formats uses binary columnar storage with embedded schema metadata?
- **A)** `.csv`
- **B)** `.json`
- **C)** `.parquet`
- **D)** `.tsv`

#### 6. What does `df.shape` return in Pandas?
- **A)** A list of column names
- **B)** A single integer representing total cells
- **C)** A Python tuple of `(rows, columns)`
- **D)** A dictionary of data types

#### 7. Why are NumPy array operations significantly faster than Python list operations?
- **A)** NumPy arrays are stored in contiguous C-level memory and use CPU SIMD vectorization
- **B)** Python lists are written in assembly language
- **C)** NumPy converts all data into string text
- **D)** Python lists use multi-threading by default

---

### 🔑 Answer Key for Section III (MCQs)

| Question # | Correct Option | Brief Technical Justification |
| :---: | :---: | :--- |
| **1** | **B) Veracity** | Veracity represents the truthfulness, cleanliness, and accuracy of the dataset. |
| **2** | **C) 8 bytes** | 64 bits $\div$ 8 bits/byte = 8 bytes per cell. |
| **3** | **B) Dictionary-based integer encoding** | Categoricals map unique string values to compact 1- or 2-byte integer indices. |
| **4** | **B) `chunksize`** | `chunksize` streams the file in smaller batches, keeping memory usage low. |
| **5** | **C) `.parquet`** | Apache Parquet is a binary columnar format optimized for large-scale analytics. |
| **6** | **C) A Python tuple of `(rows, columns)`** | `df.shape` is an attribute returning the 2D dimensional tuple. |
| **7** | **A) Contiguous C-memory & SIMD** | Vectorized arrays avoid Python object wrapping and run directly on hardware registers. |

---

<br>

<div align="right">

**Asst. Prof. Hitesh Patel**  
*Department of Computer Science & Data Science*  
*Sutex Bank College of Computer Applications & Science (SBCCAS), Amroli*  
*F.Y. / T.Y. B.Sc. (Data Science & Analytics) & B.C.A.*  

</div>
