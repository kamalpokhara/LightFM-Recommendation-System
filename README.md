# Smart Retention System: Hybrid Recommender (LightFM)

This repository contains the **Recommendation Engine** component of a comprehensive E-commerce ecosystem. This project, titled **"Smart Retention System,"** aims to increase customer lifetime value by combining churn analytics with high-precision product discovery.

> **⚠️ Project Status:** In Development. This specific module handles the **RecSys** logic using LightFM. The full system involves a parallel **Churn Prediction model** and a **Django-based E-commerce frontend.**

---

## 🚀 The "Smart Retention" Strategy

In a modern e-commerce environment, retention is driven by two pillars: identifying who is likely to leave (Churn) and showing them exactly what they want to buy (RecSys). This repository implements the discovery layer through two distinct sections:

### Section 1: "Our Popular Products" (Global)
* **Target:** All visitors (including logged-out guests).
* **Logic:** A non-ML weighted aggregation of the interactions table.
* **Weighted Scoring:** * **Purchase:** × 5
    * **Add to Cart:** × 3
    * **Wishlist:** × 2
    * **View:** × 1
* **Implementation:** Results are computed via batch jobs and cached for zero-latency serving.

### Section 2: "Recommended for You" (Personalized)
* **Target:** Logged-in users with a behavioral history.
* **Core Engine:** **LightFM** (Hybrid Matrix Factorization).
* **Retention Logic:** * Excludes previously purchased items to encourage new category discovery.
    * **Cold Start Handling:** Utilizes **Item Features** (category, brand, rating) to provide high-quality suggestions even for users with minimal history.

---

## 📊 Data Architecture & Mapping

The recommendation matrices are constructed from three primary data streams:

| CSV Source | Model Mapping | Purpose |
| :--- | :--- | :--- |
| `synthetic_interactions.csv` | **Interaction Matrix** | Behavioral history weighted by intent/action. |
| `products.csv` | **Item Feature Matrix** | Product metadata (Category, Brand, Price Range, Rating). |
| `synthetic_users.csv` | **User Feature Matrix** | Demographic data (Gender, Age Group, Interests). |

---

## 🛠️ Technical Stack & Environment

This module is optimized for a Linux-based high-performance environment to support the heavy mathematical computations required by the LightFM C-extensions.

* **OS:** Ubuntu (WSL2)
* **Language:** Python 3.10.11
* **Core Libraries:** * `LightFM`: Hybrid recommendation algorithms.
    * `Pandas/SciPy`: Sparse matrix construction and data processing.
    * `Scikit-Learn`: Evaluation metrics and preprocessing.
* **Architecture:** Designed as a microservice served via **FastAPI** to connect seamlessly with the **Django** E-commerce frontend.

---

## 🚧 Project Roadmap
- [x] Environment Configuration (WSL2 + Python 3.10)
- [x] Compilation of C-extensions (GCC/Build-Essential)
- [x] Data Pre-processing & Sparse Matrix Logic
- [ ] Final Model Training & Hyperparameter Tuning
- [ ] **Integration:** Merging RecSys + Churn Model into a unified API.
- [ ] **Deployment:** Serving real-time recommendations to the Django E-commerce platform.
