# SC1015-Data-Science-and-Artificial-Intelligence

**SC1015 - Data Science Project (Predicting Student Academic Performance)**

Filename: `README.md` (in the root of this project's GitHub repository)

```markdown
# Predicting Student Academic Performance (SC1015 Data Science Project)

This project, undertaken for the SC1015 Introduction to Data Science & Artificial Intelligence course at Nanyang Technological University, focuses on predicting students' final academic grades. It involves analyzing datasets of secondary students' performance in Math and Portuguese courses to identify key influencing factors and build predictive models.

## Table of Contents
* [Project Goal](#project-goal)
* [Dataset](#dataset)
* [Methodology](#methodology)
    * [Data Preprocessing](#data-preprocessing)
    * [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    * [Feature Selection](#feature-selection)
    * [Model Development (Conceptual Overview)](#model-development-conceptual-overview)
    * [Evaluation](#evaluation)
* [Technologies Used](#technologies-used)
* [File Structure](#file-structure)
* [How to Run](#how-to-run)
* [Key Findings & Results](#key-findings--results)
* [Author](#author)

## Project Goal

To explore factors affecting student academic performance and to develop a model capable of predicting final grades (G3) based on demographic, social, and school-related attributes.

## Dataset

The project utilizes two datasets representing student performance in:
1.  Mathematics (`student-mat.csv`) - 395 students, 33 attributes.
2.  Portuguese language (`student-por.csv`) - 649 students, 33 attributes.

Attributes include student grades (G1, G2, G3), demographic data (sex, age, address), social factors (family relationships, free time, alcohol consumption), and school-related information (study time, failures, school support).

## Methodology

### Data Preprocessing
*   Renamed columns for clarity.
*   Handled categorical features using One-Hot Encoding (`pd.get_dummies`).
*   Scaled numerical features using `StandardScaler` from Scikit-learn.

### Exploratory Data Analysis (EDA)
*   Analyzed distributions of numerical and categorical variables using histograms, box plots, and count plots (Seaborn, Matplotlib).
*   Investigated skewness and identified potential outliers in numerical features.
*   Visualized relationships between predictor variables and the target variable (`FinalGrade`).

### Feature Selection
A combination of statistical tests and correlation analysis was used to identify relevant features for predicting `FinalGrade`:
*   **Numerical Features:** Pearson correlation with `FinalGrade`. Features with correlation >= |0.2| were initially considered (e.g., `SecondGrade`, `Failures`, `MotherEdu`).
*   **Categorical Features:** ANOVA F-test (p-value <= 0.05) to assess the relationship with `FinalGrade`.
*   **Inter-categorical Dependency:** Chi-Square test (p-value < 0.05) to identify dependencies among selected categorical features, guiding further refinement.
*   **Final Selected Features (Example for Math dataset):** `Address`, `Paid`, `Higher`, `Romantic`, `SecondGrade`, `Failures`, `MotherEdu`.
*   **Final Selected Features (Example for Portuguese dataset):** `Higher`, `Internet`, `SecondGrade`, `Failures`, `MotherEdu`, `WorkdayAlc`, `StudyTime`.

### Model Development (Conceptual Overview & My Contribution)
While the final neural network models were developed by the team using TensorFlow/Keras, my role focused on preparing the data for these models and analyzing their outputs.
*   **Model Architecture (Team-Developed):** Sequential Neural Networks with Dense layers (ReLU activation for hidden layers, Linear activation for the output layer for regression).
*   **Compilation:** Adam optimizer, Mean Squared Error (MSE) loss.
*   **Training:** Trained for 100 epochs with a batch size.

My primary contributions to the modeling phase were:
*   Ensuring the data pipeline (preprocessing, feature selection) fed clean, relevant data to the models.
*   Analyzing training history (MSE, MAE, loss vs. epochs) to understand model convergence and identify potential overfitting/underfitting from the team's model runs.

### Evaluation
*   Models were evaluated using Mean Absolute Error (MAE) and Mean Squared Error (MSE) on a held-out test set (20% split).
*   A baseline MAE/MSE was established by predicting the mean of `FinalGrade`.

## Technologies Used

*   **Language:** Python 3.x
*   **Core Libraries:**
    *   Pandas (Data manipulation and analysis)
    *   NumPy (Numerical operations)
    *   Seaborn & Matplotlib (Data visualization)
    *   Scikit-learn (Data preprocessing, train-test split, metrics)
    *   TensorFlow with Keras API (for understanding team's model structure and evaluating performance)
    *   SciPy (for statistical tests like ANOVA, Chi-Square)
*   **Development Environment:** Jupyter Notebook

## File Structure

*   `Mini Project (2).md` / `Mini Project (2).ipynb`: Jupyter Notebook containing all analysis, preprocessing, model evaluation steps.
*   `student-mat.csv`: Dataset for mathematics student performance.
*   `student-por.csv`: Dataset for Portuguese student performance.

## How to Run

1.  Ensure all dependencies listed under "Technologies Used" are installed.
    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn tensorflow scipy
    ```
2.  Open the `Mini Project (2).ipynb` (or the .md converted to .ipynb) in Jupyter Notebook or Jupyter Lab.
3.  Run the cells sequentially to replicate the analysis and observe the results.

## Key Findings & Results

*   **Significant Predictors:** Features like `SecondGrade` (previous grades), `Failures`, `MotherEdu`, and `Higher` (aspiration for higher education) consistently showed strong predictive power for `FinalGrade`.
*   **Model Performance (Portuguese Dataset - Best Team Model):**
    *   Test MAE: **0.846**
    *   Test MSE: **1.563**
    *   This significantly outperformed the baseline mean prediction (MAE: ~2.40).
*   **Model Performance (Math Dataset - Best Team Model):**
    *   Test MAE: **1.762**
    *   Test MSE: **7.922**
*   The project demonstrated the effectiveness of a systematic data science pipeline, from EDA and feature selection to model evaluation, in tackling a real-world prediction problem.

## Author

*   Javier
*   Faith
*   Zhe Wei
---
