<div align="center">

# 🐘 NoSQL Assignment 2: Hadoop MapReduce Analysis

![Hadoop](https://img.shields.io/badge/Apache_Hadoop-66CCFF?style=for-the-badge&logo=apachehadoop&logoColor=black)
![Java](https://img.shields.io/badge/Java-ED8B00?style=for-the-badge&logo=openjdk&logoColor=white)
![LaTeX](https://img.shields.io/badge/LaTeX-008080?style=for-the-badge&logo=latex&logoColor=white)

*A comprehensive exploration of **Co-occurrence Analysis** and **Document Indexing** using Hadoop MapReduce paradigms.*

</div>

---

## 🎯 Project Overview

This repository contains the MapReduce implementation, runtime analysis, and execution documentation for NoSQL Assignment 2. The project focuses on building scalable algorithms to solve two main challenges:
1. **Co-occurring Word Matrix Generation** evaluating **Pairs** vs. **Stripes** approaches, and the effects of **Local Aggregation**.
2. **Indexing Documents** over massive text datasets leveraging the Hadoop ecosystem.

---

## 📂 Repository Structure

| Directory / File | Description |
|------------------|-------------|
| 💻 **[`Assig2/`](./Assig2)** | Main source code containing Java MapReduce implementations.<br> - `Prob1/`: Word Matrix Generation (Pairs, Stripes).<br> - `Prob2/`: Document Indexing. |
| 💾 **[`Solution/`](./Solution)** | Generated output files after running the MapReduce jobs.<br> - `Problem1/`: Ranked word counts and co-occurrence matrices.<br> - `Problem2/`: TF-IDF scores and document frequency outputs. |
| 📈 **[`RunTimeAnalysis/`](./RunTimeAnalysis)** | Detailed performance metrics, scalability data, and comparative analysis (Testing vs. Large datasets). |
| 📸 **[`ScreenShots/`](./ScreenShots)** | Visual execution logs, Hadoop Namenode UI views, and graphical output verifications. |
| ⚙️ **[`Problem_1.txt`](./Problem_1.txt) / [`Problem_2.txt`](./Problem_2.txt)** | Step-by-step compilation and execution commands for the Hadoop cluster. |
| 📝 **[`report.tex`](./report.tex)** | The comprehensive LaTeX final report document. |
| 📋 **[`NoSQL_Assignment2.pdf`](./NoSQL_Assignment2.pdf)** | Original assignment specification and exact problem statements. |

> **Note:** The `Dataset/` folder has been intentionally excluded from Git tracking due to GitHub's file size limits, but the structure is required locally for execution.

---

## 🛠️ How to Run

Detailed compilation and execution instructions for the Hadoop ecosystem can be found in the root documentation files:
- 👉 See **[`Problem_1.txt`](./Problem_1.txt)** for compiling and running the Co-occurrence Matrix code.
- 👉 See **[`Problem_2.txt`](./Problem_2.txt)** for compiling and running the Document Indexing code.

---

## 📊 Performance Analysis

The `RunTimeAnalysis` documents critically evaluate the efficiency of the developed MapReduce algorithms. Key analytical points include:

- ⚖️ **Pairs vs. Stripes:** Evaluating the memory overhead of Stripes versus the sorting overhead of Pairs.
- ⚡ **Local Aggregation:** Assessing network shuffle reduction using In-Mapper Combiners.
- 📈 **Scalability Testing:** Comparing runtime differences across small verification sets and large-scale real-world data.

---

<div align="center">
  <i>🎓 Developed as part of the NoSQL Systems course (Semester 6).</i>
</div>
