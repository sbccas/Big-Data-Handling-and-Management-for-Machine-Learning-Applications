# 💡 Unit 1 Question Bank, Glossary & Viva Voce Exam Vault

> **Course Code:** DS-505 | **Subject:** Big Data Handling and Management for Machine Learning Applications  
> **Course Degree:** B.Sc. (Data Science & Analytics) — Semester V  
> **Institution:** Sutex Bank College of Computer Applications and Science (SBCCAS), Amroli  
> **Affiliation:** Veer Narmad South Gujarat University (VNSGU), Surat  
> **Unit Covered:** Unit 1 — Introduction to Big Data and Large Dataset Handling  
> **Document Purpose:** Comprehensive Exam Preparation, Quick Revision & Practical Viva Voce Guide  
> **Lecture Note Reference:** [Unit-1_Introduction_to_Big_Data_and_Large_Dataset_Handling.md](../2_Lecture_Notes/Unit-1_Introduction_to_Big_Data_and_Large_Dataset_Handling.md)  

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
| **1** | **Big Data** | Datasets characterized by high Volume, Velocity, Variety, Veracity, and Value that cannot be stored, managed, or processed using traditional single-node RDBMS systems. |
| **2** | **Volume** | The physical scale of data generation and storage, typically quantified in Terabytes ($10^{12}$), Petabytes ($10^{15}$), and Exabytes ($10^{18}$). |
| **3** | **Velocity** | The high speed at which new data is generated, transmitted, and required to be processed into actionable intelligence in real-time or near-real-time. |
| **4** | **Variety** | The multi-structural nature of modern datasets across structured (SQL), semi-structured (JSON/XML), and unstructured (video/audio/text) formats. |
| **5** | **Veracity** | The degree of reliability, quality, and trustworthiness of data, requiring cleaning to remove noise, anomalies, and missing values. |
| **6** | **Value** | The actionable business insight or predictive utility extracted from raw data through statistical modeling and machine learning. |
| **7** | **Vertical Scaling** | Upgrading a single server by adding more hardware resources such as RAM, faster CPUs, or bigger hard drives ("scale-up"). |
| **8** | **Horizontal Scaling** | Connecting multiple commodity servers together in a cluster to share the storage and computational processing load ("scale-out"). |
| **9** | **Schema-on-Read** | A storage pattern where raw data is stored without a rigid structure, and schema is applied dynamically only when reading the data. |
| **10** | **Vectorization** | Performing mathematical operations across entire arrays simultaneously using compiled C-level SIMD instructions, eliminating slow Python `for` loops. |
| **11** | **SIMD** | *Single Instruction, Multiple Data*; a CPU hardware capability that executes a single instruction across multiple data points in parallel. |
| **12** | **Chunking** | Partitioning a large disk file into smaller row subsets to stream through system RAM incrementally via the `chunksize` parameter in Pandas. |
| **13** | **Downcasting** | Converting high-precision data types (e.g., `float64`, `int64`) to smaller representations (e.g., `float32`, `int16`) to conserve system RAM. |
| **14** | **Categorical Dtype** | A Pandas memory optimization technique that encodes repetitive string columns into integer codes with an internal dictionary lookup. |
| **15** | **EDA** | *Exploratory Data Analysis*; the systematic methodology of summarizing main dataset characteristics through numerical and visual diagnostics. |

---

## 2. Short Answer Theory Questions (2 to 3 Marks Each)

#### Q1: Differentiate between Horizontal Scaling and Vertical Scaling.
> **Model Answer:**  
> * **Vertical Scaling (Scale-Up):** Increasing computing capacity by adding more RAM, faster CPUs, or bigger disks to a **single machine**. *Limitation:* It has an absolute physical hardware ceiling and becomes prohibitively expensive.  
> * **Horizontal Scaling (Scale-Out):** Increasing computing capacity by adding **more commodity computers (nodes)** into a distributed cluster (e.g., Hadoop, Spark). *Advantage:* Highly cost-effective and provides virtually unlimited scale.

#### Q2: What causes the Out-Of-Memory (OOM) error in Pandas when loading large files?
> **Model Answer:**  
> The OOM error occurs because of **Memory Amplification**:
> 1. Pandas defaults to 64-bit numerical types (`int64`, `float64`), consuming 8 bytes per cell regardless of value size.
> 2. Text columns default to `object` dtypes, storing individual 64-bit memory pointers to fragmented Python string objects.
> 3. A CSV file occupying 1 GB on disk often requires 3 GB to 5 GB of RAM when unpacked into memory, exceeding system limits.

#### Q3: What is the functional role of the `chunksize` parameter in `pd.read_csv()`?
> **Model Answer:**  
> When `chunksize=N` is specified, `pd.read_csv()` does not read the entire file at once into a DataFrame. Instead, it returns an **iterable `TextFileReader` object** that yields DataFrames of $N$ rows per iteration. This allows data scientists to process arbitrarily large files out-of-core on low-RAM machines.

#### Q4: Why is Apache Parquet preferred over CSV for Big Data storage?
> **Model Answer:**  
> 1. **Columnar Storage:** Parquet reads only the columns required by a query, whereas CSV must scan every byte on disk.
> 2. **High Compression:** Built-in Snappy/Gzip compression reduces disk footprints by up to 80%.
> 3. **Preserves Schema:** Embedded metadata preserves exact data types, avoiding parsing errors on load.

#### Q5: What is the difference between `df.dropna(how='any')` and `df.dropna(how='all')`?
> **Model Answer:**  
> * `df.dropna(how='any')`: Drops a row if **at least one** column contains a missing value (`NaN`).
> * `df.dropna(how='all')`: Drops a row **only if all** columns in that row are simultaneously `NaN`.

---

## 3. Long Answer Comprehensive Theory Questions (5 to 7 Marks Each)

#### Q1: Explain the 5 Vs of Big Data with neat illustrative examples. Discuss the limitations of traditional databases in managing them.
* **Key Points to Cover:**
  - Define Big Data (Gartner / McKinsey definition).
  - Detailed breakdown of **Volume**, **Velocity**, **Variety**, **Veracity**, and **Value**.
  - Draw the 5Vs architecture diagram.
  - Comparative table: Traditional RDBMS (MySQL/Oracle) vs. Modern Big Data Systems (Hadoop/Spark).

#### Q2: Explain the Python memory model when handling large datasets. Detail three proven techniques to optimize RAM consumption in Pandas.
* **Key Points to Cover:**
  - Concept of Memory Amplification (Python `PyObject` overhead and 64-bit pointer arrays).
  - **Technique 1:** Downcasting floating-point (`float64` &rarr; `float32`) and integers (`int64` &rarr; `int16`/`int32`).
  - **Technique 2:** Categorical dictionary encoding for repetitive strings (`object` &rarr; `category`).
  - **Technique 3:** Selective column ingestion using `usecols`.
  - Include code snippet showing how memory drops by 80%+.

#### Q3: Compare Structured, Semi-Structured, and Unstructured Data. Provide industrial use cases and storage solutions for each.
* **Key Points to Cover:**
  - Comprehensive 3-column table comparing structure, schema rigidity, flexibility, and scalability.
  - Real-world industrial examples (RDBMS for banking transactions, JSON for web APIs and NoSQL, Video/Audio for computer vision and speech AI).

#### Q4: Detail the architecture and operation of Chunked Data Processing using Pandas. Explain how global aggregates (mean, sum, count) are computed across chunks.
* **Key Points to Cover:**
  - Draw sequence diagram of Disk &rarr; RAM Chunk &rarr; Accumulator.
  - Code walkthrough initializing global counters (`total_sum`, `total_count`).
  - Explain why mathematical operations like median require specialized algorithms across chunks.

---

## 4. Practical Examination Viva Voce Questions & Model Answers

1. **Q: Does `df.drop('col_name', axis=1)` modify the original DataFrame by default?**  
   *A:* No. By default, it returns a new modified DataFrame copy. To mutate the existing DataFrame in place, you must pass `inplace=True`.
2. **Q: Why should you pass `memory_usage='deep'` when calling `df.info()`?**  
   *A:* Without `deep=True`, Pandas reports estimated memory based on pointer arrays alone. `deep=True` forces Pandas to walk through heap memory and inspect the real bytes allocated to string objects.
3. **Q: What is the output of `df.shape` on a dataset with 50,000 rows and 12 columns?**  
   *A:* A Python tuple: `(50000, 12)`.
4. **Q: Can `np.nan` be compared directly using `x == np.nan`?**  
   *A:* No. IEEE floating-point standards dictate that `NaN` is not equal to anything, including itself. You must use `np.isnan(x)` or `pd.isna(x)`.
5. **Q: When reading a JSON file with one JSON document per line, which argument must be set in `pd.read_json()`?**  
   *A:* `lines=True`.
6. **Q: How does `df.fillna(method='ffill')` work in time-series data?**  
   *A:* It performs a "forward fill", propagating the last non-null valid observation forward to fill subsequent `NaN` cells.
7. **Q: What is the difference between `df['col']` and `df[['col']]`?**  
   *A:* `df['col']` returns a 1-dimensional **Pandas Series**, while `df[['col']]` returns a 2-dimensional **Pandas DataFrame** containing one column.
8. **Q: Why is NumPy faster than pure Python lists for numerical operations?**  
   *A:* NumPy stores data in contiguous blocks of memory with homogeneous C data types and executes operations via low-level compiled SIMD machine instructions without Python interpreter overhead.
9. **Q: What is the return type of `pd.read_csv('file.csv', chunksize=10000)`?**  
   *A:* It returns a `pandas.io.parsers.TextFileReader` iterator object, not a DataFrame.
10. **Q: What is the difference between `int64` and `int8` in terms of memory consumption?**  
    *A:* `int64` consumes 8 bytes (64 bits) per value, whereas `int8` consumes only 1 byte (8 bits) per value—yielding an 87.5% memory reduction per element.

---

<div align="center">

**Department of Computer Science & Data Science**  
**Sutex Bank College of Computer Applications and Science (SBCCAS), Amroli**  
*Veer Narmad South Gujarat University (VNSGU), Surat*

</div>
