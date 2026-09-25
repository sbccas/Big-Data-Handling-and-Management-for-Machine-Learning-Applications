<div align="center">

# SUTEX BANK COLLEGE OF COMPUTER APPLICATIONS & SCIENCE, AMROLI
### T.Y.B.Sc. (DATA SCIENCE & ANALYTICS) — SEMESTER: 5th
## ASSIGNMENT – II (UNIT – II)
### SUBJECT: DS-505 — Big Data Handling and Management for Machine Learning Applications

**Unit Covered:** Unit 2 – Big Data Processing using PySpark  
**Submission Date:** *To be announced by Course Instructor*  
**Recommended Student Notebook Length:** **12 to 15 Pages Total**  

---

</div>

> **Instructions for Students:**
> 1. Handwrite all answers neatly in your official course assignment notebook / sheets.
> 2. Each **Long Question** should take **~2 to 2.5 pages** (including architecture diagrams, execution trees, and comparison tables).
> 3. Each **Short Question** should take **~0.5 to 0.75 page**.
> 4. For **Multiple Choice Questions (MCQs)**, write the question number, correct option letter, and full option text.
> 5. Total assignment volume should be between **12 to 15 handwritten pages**.
> 6. All questions are directly searchable and referenced from your lecture notes:  
>    👉 [**`2_Lecture_Notes/Unit-2_Big_Data_Processing_using_PySpark.md`**](../2_Lecture_Notes/Unit-2_Big_Data_Processing_using_PySpark.md)

---

## SECTION I: Answer in Detail (Long Questions)
*(Attempt all questions. Expected length: ~2 to 2.5 handwritten pages per question)*

### 1. Explain the Distributed Computing Paradigm, Scale-Up vs. Scale-Out, and Why Apache Spark Outperforms Hadoop MapReduce.
* Discuss the physical limitations of single-node computing (RAM ceiling, CPU core limits, I/O bus contention, OOM crashes).
* Compare **Vertical Scaling (Scale-Up)** versus **Horizontal Scaling (Scale-Out)** across cost curves, physical limits, fault tolerance, and software complexity.
* Detail the fundamental architectural bottleneck of **Hadoop MapReduce** (multi-pass disk I/O between Map and Reduce phases).
* Explain how **Apache Spark's in-memory computing** and Resilient Distributed Datasets (RDDs) achieve up to 100x performance improvements.
* Draw the comparative workflow diagram illustrating MapReduce disk cycles versus Spark in-memory pipelines.  
*(📖 **Reference:** Lecture Notes `Section 1.1`, `Section 1.2`, & `Section 1.3`)*

---

### 2. Detail the Apache Spark Distributed Architecture with a Labeled Architectural Diagram.
* Define the Master-Worker distributed topology in Apache Spark.
* Explain the role, internal components, and responsibilities of:
  1. **The Driver Node:** `SparkSession`, `DAGScheduler`, `TaskScheduler`, and task dispatching.
  2. **The Cluster Manager:** Role of resource allocation across Standalone, Apache YARN, and Kubernetes.
  3. **Worker Nodes & Executors:** Dedicated JVM processes, CPU core threading, RAM cache storage, and task execution.
* Explain the execution hierarchy: **Application $\rightarrow$ Job $\rightarrow$ Stage $\rightarrow$ Task**.
* Draw the complete, labeled Apache Spark Distributed Architecture diagram.  
*(📖 **Reference:** Lecture Notes `Section 2.1`, `Section 2.2`, & `Section 2.3`)*

---

### 3. Explain the Concepts of Lazy Evaluation, Directed Acyclic Graph (DAG), and Narrow vs. Wide Dependencies.
* Define **Lazy Evaluation** and explain why Spark defers physical execution until an Action is invoked.
* What is a **Directed Acyclic Graph (DAG)**? Explain its mathematical and operational properties in Spark.
* Differentiate between **Transformations** and **Actions** with examples of each.
* Thoroughly explain the difference between:
  * **Narrow Dependencies** (Pipelineable operations, e.g., `select()`, `filter()`, zero network shuffle).
  * **Wide Dependencies** (Shuffle operations, e.g., `groupBy()`, `join()`, network redistribution across partitions).
* Draw the execution pipeline showing how an Action triggers DAG decomposition into Stages across shuffle boundaries.  
*(📖 **Reference:** Lecture Notes `Section 3.2`, `Section 3.3`, & `Section 3.4`)*

---

### 4. Explain PySpark Architecture, the Py4J Gateway, and Initializing the SparkSession using the Builder Pattern.
* Explain how Python scripts interact with the Scala-based JVM Spark engine via the **Py4J gateway**.
* Why do modern PySpark DataFrames execute at near-identical speed to native Scala code (Off-heap memory and the Catalyst Optimizer)?
* Discuss the role of the modern unified **`SparkSession`** compared to legacy `SparkContext` and `SQLContext`.
* Write a clean, complete PySpark script demonstrating `SparkSession.builder` initialization with:
  * Application Name (`appName`)
  * Master mode (`local[*]`)
  * Driver memory allocation (`spark.driver.memory`)
  * Shuffle partition tuning (`spark.sql.shuffle.partitions`)
* Explain what each configuration parameter does.  
*(📖 **Reference:** Lecture Notes `Section 4.1`, `Section 4.2`, & `Section 4.3`)*

---

### 5. Detail the Six Mandatory Syllabus PySpark DataFrame Operations with Syntax, Parameters, and Execution Code.
* For each of the following six mandatory syllabus functions, detail its:
  * **Operational Type** (Transformation vs. Action)
  * **Dependency Type** (Narrow vs. Wide, where applicable)
  * **Method Signature & Important Parameters**
  * **Practical Big Data Use Case**
  1. `df.show()` (Parameters: `n`, `truncate`, `vertical`)
  2. `df.printSchema()` (Tree hierarchy of column names and datatypes)
  3. `df.select()` (Column projection, renaming, and mathematical calculations)
  4. `df.filter()` / `df.where()` (SQL syntax vs. Pythonic bitwise conditions)
  5. `df.groupBy()` (Grouping keys and aggregate calculations with `.agg()`)
  6. `df.count()` (Distributed cardinality calculation across partitions)
* Provide a single, complete, unified PySpark script demonstrating all six operations chained on a sample dataset.  
*(📖 **Reference:** Lecture Notes `Section 6.1` to `Section 6.7`)*

---

## SECTION II: Answer Briefly (Short Questions)
*(Attempt all questions. Expected length: ~0.5 to 0.75 handwritten page per question)*

### 1. Differentiate between Narrow Dependencies and Wide Dependencies in Apache Spark.
* Explain partition mapping (1-to-1 vs. Many-to-Many) and why wide dependencies require a network shuffle.  
*(📖 **Reference:** Lecture Notes `Section 3.4`)*

---

### 2. Why is Apache Parquet Preferred over Plain Text CSV in PySpark Big Data Workloads?
* Discuss columnar storage, projection pruning, embedded metadata, and built-in Snappy compression.  
*(📖 **Reference:** Lecture Notes `Section 5.4`)*

---

### 3. Differentiate between `df.cache()` and `df.persist()` in PySpark.
* Explain default storage levels (`MEMORY_AND_DISK`) and when a data scientist should choose `MEMORY_ONLY_SER`.  
*(📖 **Reference:** Lecture Notes `Section 7.1`)*

---

### 4. What is the Catastrophic Danger of Calling `df.toPandas()` or `df.collect()` on Large Distributed Datasets?
* Explain the physical mechanism of driver memory saturation and Out-Of-Memory (OOM) heap crashes.  
*(📖 **Reference:** Lecture Notes `Section 7.3`)*

---

### 5. Differentiate between `df.repartition()` and `df.coalesce()`.
* Explain which method performs a full network shuffle and which should be used to reduce output partitions efficiently before disk writes.  
*(📖 **Reference:** Lecture Notes `Section 7.2`)*

---

## SECTION III: Multiple Choice Questions (MCQs)
*(Choose the single correct option. Write the question number, option letter, and answer text).*

#### 1. In Apache Spark architecture, which component is responsible for running the application's `main()` function and constructing the DAG?
- **A)** Worker Node
- **B)** Driver Program
- **C)** Cluster Manager
- **D)** Executor JVM

#### 2. What type of operation is `df.groupBy()` in Apache Spark?
- **A)** Action
- **B)** Narrow Transformation
- **C)** Wide Transformation (Requires Shuffle)
- **D)** Metadata Utility

#### 3. Which bridge library enables Python to interact directly with Java Virtual Machine (JVM) objects in PySpark?
- **A)** PyArrow
- **B)** Cython
- **C)** Py4J
- **D)** Swig

#### 4. Which of the following operations is an **Action** that triggers physical cluster job execution?
- **A)** `df.filter()`
- **B)** `df.select()`
- **C)** `df.drop()`
- **D)** `df.count()`

#### 5. When specifying `.master("local[*]")` in a SparkSession builder, what does `*` signify?
- **A)** Run indefinitely without timeout
- **B)** Utilize all available logical CPU cores on the local machine
- **C)** Connect to all available cloud clusters
- **D)** Allocate maximum available physical RAM

#### 6. Which query optimization component in Spark SQL automatically performs Predicate Pushdown and Projection Pruning?
- **A)** Project Tungsten
- **B)** Catalyst Optimizer
- **C)** DAG Scheduler
- **D)** Task Manager

#### 7. If you need to decrease the number of partitions in a DataFrame from 200 down to 4 without causing a full network shuffle, which method is optimal?
- **A)** `df.repartition(4)`
- **B)** `df.coalesce(4)`
- **C)** `df.partitionBy(4)`
- **D)** `df.shrink(4)`

---

### 🔑 Answer Key for Section III (MCQs)

| Question # | Correct Option | Brief Technical Justification |
| :---: | :---: | :--- |
| **1** | **B) Driver Program** | The Driver hosts `main()`, creates the `SparkSession`, and schedules the DAG. |
| **2** | **C) Wide Transformation** | `groupBy()` groups keys across different partitions, requiring an expensive network shuffle. |
| **3** | **C) Py4J** | Py4J provides dynamic gateway sockets between Python and JVM runtime objects. |
| **4** | **D) `df.count()`** | `count()` returns an integer value to the driver, triggering immediate cluster job execution. |
| **5** | **B) Utilize all available logical CPU cores** | `local[*]` instructs Spark to spawn worker threads matching the system's logical cores. |
| **6** | **B) Catalyst Optimizer** | The Catalyst engine converts logical queries into optimized physical execution plans. |
| **7** | **B) `df.coalesce(4)`** | `coalesce()` merges adjacent partitions locally without triggering a full network shuffle. |

---

<br>

<div align="right">

**Asst. Prof. Hitesh Patel**  
*Department of Computer Science & Data Science*  
*Sutex Bank College of Computer Applications & Science (SBCCAS), Amroli*  
*F.Y. / T.Y. B.Sc. (Data Science & Analytics) & B.C.A.*  

</div>
