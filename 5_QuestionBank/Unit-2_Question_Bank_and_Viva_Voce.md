# 💡 Unit 2 Question Bank, Glossary & Viva Voce Exam Vault

> **Course Code:** DS-505 | **Subject:** Big Data Handling and Management for Machine Learning Applications  
> **Course Degree:** B.Sc. (Data Science & Analytics) — Semester V  
> **Institution:** Sutex Bank College of Computer Applications and Science (SBCCAS), Amroli  
> **Affiliation:** Veer Narmad South Gujarat University (VNSGU), Surat  
> **Unit Covered:** Unit 2 — Big Data Processing using PySpark  
> **Document Purpose:** Comprehensive Exam Preparation, Quick Revision & Practical Viva Voce Guide  
> **Lecture Note Reference:** [Unit-2_Big_Data_Processing_using_PySpark.md](../2_Lecture_Notes/Unit-2_Big_Data_Processing_using_PySpark.md)  

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
| **1** | **Apache Spark** | An open-source, distributed general-purpose cluster computing system designed for fast, in-memory processing of large-scale big data workloads. |
| **2** | **Driver Program** | The central controller process of a Spark application that runs the `main()` function, initializes `SparkSession`, creates the DAG, and coordinates task execution. |
| **3** | **Cluster Manager** | An external resource management service (such as Standalone, Apache YARN, or Kubernetes) responsible for acquiring and allocating computing resources across cluster nodes. |
| **4** | **Worker Node** | A physical or virtual slave machine in the cluster that runs executor processes and provides CPU cores and RAM for task execution. |
| **5** | **Executor** | A dedicated Java Virtual Machine (JVM) process launched on a worker node that executes computational tasks and stores cached data partitions in memory or disk. |
| **6** | **Task** | The smallest atomic unit of work in Apache Spark, representing the execution of a stage's computational logic on a single partition of data on a single CPU core. |
| **7** | **SparkSession** | The unified single entry point introduced in Spark 2.0 to program with Spark's DataFrame and Dataset APIs using the Builder pattern. |
| **8** | **Spark DataFrame** | A distributed, in-memory collection of data organized into named columns, equipped with a schema and optimized by the Catalyst Optimizer. |
| **9** | **Lazy Evaluation** | An execution strategy where Spark defers physical computation until an Action is explicitly called, recording transformations in a lineage graph to enable query optimization. |
| **10** | **Directed Acyclic Graph (DAG)** | A finite directed graph with no directed cycles that represents the complete sequence of transformation dependencies and execution stages in a Spark job. |
| **11** | **Transformation** | An operation on a DataFrame (e.g., `select()`, `filter()`) that produces a new DataFrame lazily without triggering immediate cluster execution. |
| **12** | **Action** | An operation on a DataFrame (e.g., `count()`, `show()`, `collect()`) that triggers the physical execution of the DAG and returns a non-DataFrame result or persists data. |
| **13** | **Narrow Dependency** | A transformation where each partition of the parent DataFrame is utilized by at most one child partition, requiring zero network data transfer (e.g., `filter()`). |
| **14** | **Wide Dependency (Shuffle)** | A transformation where multiple child partitions depend on data across multiple parent partitions, requiring an expensive cluster network data redistribution (e.g., `groupBy()`). |
| **15** | **Catalyst Optimizer** | Spark SQL’s internal rule-based and cost-based query optimization engine that performs predicate pushdown, projection pruning, and whole-stage code generation. |

---

## 2. Short Answer Theory Questions (2 to 3 Marks Each)

#### Q1: What is the primary operational difference between the Driver and an Executor in Apache Spark?
> **Model Answer:**  
> * **Driver:** Acts as the **brain / master** of the application. It runs the `main()` program, manages the `SparkSession`, converts user code into a DAG of execution stages, and schedules tasks. It does **not** perform heavy row-level data processing.  
> * **Executor:** Acts as the **muscle / worker**. It is a dedicated JVM process on a worker node that executes individual tasks assigned by the driver, keeps data cached in memory, and returns execution metrics to the driver.

#### Q2: Differentiate between Narrow and Wide Dependencies with examples.
> **Model Answer:**  
> * **Narrow Dependency:** Each partition of the parent DataFrame is used by at most one partition of the child DataFrame. Execution is local to the worker node and requires **no network shuffle** (e.g., `df.select()`, `df.filter()`).  
> * **Wide Dependency:** Multiple child partitions depend on data from multiple parent partitions. Data must be sorted and transferred across the network in an expensive **Shuffle operation** (e.g., `df.groupBy()`, `df.join()`).

#### Q3: Why is Lazy Evaluation considered a major performance advantage in Spark?
> **Model Answer:**  
> 1. **Catalyst Optimization:** Spark delays execution until an Action is called, enabling the Catalyst optimizer to analyze the entire DAG and apply **Predicate Pushdown** (filtering at storage level) and **Projection Pruning** (loading only required columns).  
> 2. **Eliminates Intermediate I/O:** Multiple narrow transformations are fused into a single in-memory pass, preventing intermediate writes to disk.  
> 3. **Fault Tolerance:** If a partition is lost, Spark recomputes only the lost partition using its recorded DAG lineage.

#### Q4: What is the functional difference between `df.show()` and `df.printSchema()`?
> **Model Answer:**  
> * **`df.show()`**: An **Action** that triggers physical cluster job execution, retrieves the first $N$ rows (default 20), and displays them in an ASCII formatted table on the console.  
> * **`df.printSchema()`**: A **Metadata Utility** that does not execute data processing tasks; it reads the cached DataFrame catalog metadata and prints the column hierarchy, datatypes, and nullability flags in a tree format.

#### Q5: Differentiate between `df.repartition()` and `df.coalesce()`.
> **Model Answer:**  
> * **`df.repartition(n)`**: Can increase or decrease the number of partitions. It performs a **full cluster network shuffle**, redistributing data evenly across all partitions.  
> * **`df.coalesce(n)`**: Can **only decrease** partitions. It merges adjacent partitions locally on the same worker, executing with **zero network shuffle**, making it significantly faster for saving final results.

---

## 3. Long Answer Comprehensive Theory Questions (5 to 7 Marks Each)

#### Q1: Explain the Apache Spark Distributed Architecture with a neat labeled diagram and detail the responsibilities of each component.
> **Structuring Your Answer for Full Marks:**
> 1. **Introduction:** Define Spark’s Master-Worker distributed computing model.
> 2. **Diagram:** Draw the labeled Spark Distributed Architecture diagram showing Driver Node (`SparkSession`, `DAGScheduler`, `TaskScheduler`), Cluster Manager (YARN / K8s / Standalone), and Worker Nodes with Executors running CPU cores, RAM cache, and tasks.
> 3. **Driver Responsibilities:** Coordinating execution, creating the DAG, scheduling stages, and collecting results.
> 4. **Cluster Manager Responsibilities:** Resource allocation, node heartbeats, and worker health monitoring.
> 5. **Worker & Executor Responsibilities:** Task execution, memory caching (`StorageLevel`), and shuffle spill management.
> 6. **Execution Hierarchy:** Define Application $\rightarrow$ Job $\rightarrow$ Stage $\rightarrow$ Task.

#### Q2: Compare Hadoop MapReduce and Apache Spark architectures and explain why Spark is up to 100x faster.
> **Structuring Your Answer for Full Marks:**
> 1. **Core Architectural Difference:** Explain MapReduce’s disk-bound architecture versus Spark’s memory-first DAG execution.
> 2. **The Disk Bottleneck in MapReduce:** Explain how MapReduce writes intermediate map outputs to disk and reads them back across the network during shuffle, causing high disk I/O latency.
> 3. **Spark In-Memory RDD / DataFrame Caching:** Explain how Spark retains data in distributed RAM across iterations, making machine learning and iterative algorithms run up to 100x faster.
> 4. **DAG vs. Two-Stage Map-Reduce:** Compare Spark's flexible multi-stage graph against MapReduce's rigid 2-stage (Map $\rightarrow$ Reduce) execution.
> 5. **Comparative Table:** Contrast storage medium, speed, API language support, iterative performance, and failure recovery.

#### Q3: Explain the Catalyst Optimizer and the Complete Lifecycle of a PySpark Query.
> **Structuring Your Answer for Full Marks:**
> 1. **Introduction:** Define the Catalyst Optimizer and why it equalizes PySpark and Scala performance.
> 2. **Phase 1: Analysis:** Resolving column names and table references against the internal catalog to produce a Resolved Logical Plan.
> 3. **Phase 2: Logical Optimization:** Rule-based optimizations including Predicate Pushdown, Projection Pruning, and constant folding.
> 4. **Phase 3: Physical Planning:** Generating multiple physical execution strategies (e.g., Broadcast Hash Join vs. Sort Merge Join) and selecting the cheapest via Cost-Based Optimization (CBO).
> 5. **Phase 4: Code Generation (Project Tungsten):** Compiling physical plan operators into whole-stage Java bytecode running directly on CPU hardware registers.

#### Q4: Detail the Six Core Syllabus PySpark DataFrame Operations with Syntax, Examples, and Practical Use Cases.
> **Structuring Your Answer for Full Marks:**
> 1. Classify each function into Transformation vs. Action and Narrow vs. Wide:
>    * `df.show()`: Action, displays tabular rows.
>    * `df.printSchema()`: Metadata inspection, displays structural tree.
>    * `df.select()`: Narrow transformation, projects and computes columns.
>    * `df.filter()`: Narrow transformation, applies boolean conditions with predicate pushdown.
>    * `df.groupBy()`: Wide transformation, groups keys across partitions causing network shuffle.
>    * `df.count()`: Action, returns total row count as a native integer.
> 2. Provide clear syntax and parameter breakdown for each method.
> 3. Write a clean 10-line PySpark code snippet chaining all six operations together on a sample dataset.

---

## 4. Practical Examination Viva Voce Questions & Model Answers

Examiners frequently ask these 15 questions during university practical examinations:

#### 1. What is PySpark, and how does Python communicate with Spark’s JVM?
> **Answer:** PySpark is the Python API for Apache Spark. Python communicates with the underlying Java Virtual Machine (JVM) using the **Py4J library**, which opens a local TCP socket gateway between the Python runtime and the JVM.

#### 2. What does `.master("local[*]")` signify when building a SparkSession?
> **Answer:** It tells Spark to run locally in single-machine mode, and the asterisk `[*]` instructs Spark to utilize all available logical CPU cores on the local machine as worker threads.

#### 3. What is the difference between an Action and a Transformation in PySpark?
> **Answer:** A **Transformation** (e.g., `filter()`, `select()`) is lazy and returns a new DataFrame without executing data processing. An **Action** (e.g., `count()`, `show()`) triggers physical cluster job execution and returns a non-DataFrame result or writes data to disk.

#### 4. Why does `len(df)` fail on a PySpark DataFrame?
> **Answer:** `len()` is a Python built-in function that expects an in-memory sequence with a local size counter. Because a PySpark DataFrame is distributed across cluster worker nodes, Python cannot measure its length locally. We must use the distributed action **`df.count()`**.

#### 5. What happens under the hood when `df.groupBy()` is executed?
> **Answer:** Spark performs a **network shuffle**. Rows sharing the same grouping key are extracted from across all partitions, serialized, transmitted over the cluster network, and grouped together into reducer partitions on worker nodes.

#### 6. What is Predicate Pushdown?
> **Answer:** An optimization where filtering conditions (e.g., `col("city") == "Surat"`) are pushed directly to the underlying file storage layer (such as Parquet or JDBC), allowing Spark to load only relevant data blocks into RAM, drastically reducing memory usage and I/O.

#### 7. Why should you avoid `inferSchema=True` on production Big Data CSV files?
> **Answer:** `inferSchema=True` forces Spark to scan the entire dataset twice: first to guess column datatypes, and second to actually ingest the data. In production, we explicitly define schemas using `StructType` and `StructField` for zero overhead.

#### 8. Why is Apache Parquet significantly faster to read than CSV?
> **Answer:** Parquet is a columnar storage format with embedded metadata and built-in Snappy compression. Spark performs **Projection Pruning** (skipping unneeded columns) and **Predicate Pushdown** (skipping data blocks using min/max metadata in the footer) without scanning unneeded rows.

#### 9. What is the difference between `df.cache()` and `df.persist()`?
> **Answer:** `df.cache()` is a shortcut that uses the default storage level `MEMORY_AND_DISK`. `df.persist(StorageLevel)` allows developers to specify custom storage levels, such as `MEMORY_ONLY`, `MEMORY_ONLY_SER` (serialized bytes), or disk replication levels.

#### 10. Why is calling `df.toPandas()` or `df.collect()` dangerous on massive datasets?
> **Answer:** Both operations pull all distributed rows across every worker node into the single **Driver node's memory**. If the dataset exceeds the Driver's allocated RAM, the JVM or Python process crashes with an Out-Of-Memory (OOM) error.

#### 11. What is the difference between `df.filter()` and `df.where()`?
> **Answer:** There is zero functional difference. `df.where()` is an exact syntactic alias of `df.filter()`, provided to make SQL developers feel familiar with the DataFrame API.

#### 12. How do you combine multiple conditions in a PySpark `filter()`?
> **Answer:** You must use bitwise operators (`&` for AND, `|` for OR, `~` for NOT), and **each individual condition must be wrapped in parentheses**, e.g.:  
> `df.filter((col("age") > 18) & (col("city") == "Surat"))`

#### 13. What is the default number of shuffle partitions in Spark, and how do you configure it?
> **Answer:** The default is **200 partitions** (`spark.sql.shuffle.partitions`). On local workstations or small lab datasets, it is configured using:  
> `spark.conf.set("spark.sql.shuffle.partitions", "4")`

#### 14. What is the difference between an RDD and a DataFrame?
> **Answer:** An RDD (Resilient Distributed Dataset) contains raw, unstructured Java/Python objects without schema knowledge and cannot be optimized by Catalyst. A DataFrame is organized into named columns with a known schema, enabling the Catalyst optimizer and Project Tungsten to generate optimized off-heap bytecode.

#### 15. What is the default port for the Apache Spark Web UI?
> **Answer:** Port **`4040`** (e.g., `http://localhost:4040`). It provides real-time monitoring of active jobs, stages, storage memory, DAG execution visualizations, and executor health.

---
*(End of Unit 2 Question Bank & Viva Voce Exam Vault)*
