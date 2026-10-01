# Student Performance Prediction - Group 18

This project uses the UCI Student Performance dataset to predict each student's final mathematics grade (G3).

## Team

- Basil Yousif
- Abel Sanchez
- Group 18
- CSE 4334-002

## Dataset

The project uses `student-mat.csv`, which contains 395 student records and 33 attributes. The original dataset is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance).

## Project files

- `student_grade_prediction.ipynb`: exploratory analysis, model training, evaluation, and the actual-versus-predicted plot
- `task2_preprocessing.py`: categorical encoding, numerical standardization, and correlation analysis
- `predictions.csv`: actual and predicted test-set grades
- `actual_vs_predicted.png`: Task 4 performance graph
- `task4_results.txt`: Task 4 metrics and observations
- `group18_dataMining.pptx`: 14-slide project presentation
- `student-mat.csv`: mathematics dataset used by the project

## Method

1. Review summary statistics and visualize the grade data.
2. One-hot encode categorical columns and standardize numerical columns using training-set values.
3. Split the data into 80% training and 20% testing sets.
4. Train a Decision Tree Regressor with `random_state=42`.
5. Evaluate the test predictions using MSE and R squared.

## Results

The submitted decision tree produced:

- Mean Squared Error: **5.584**
- R squared: **0.728**

The model explains about 72.8% of the variation in final grades. In the Actual vs. Predicted Grades scatter plot, most predictions follow the dashed `y = x` line. The model overpredicts several students whose actual final grade is 0, which accounts for some of the larger errors.

## Presentation

`group18_dataMining.pptx` is a 14-slide presentation prepared by Basil Yousif and Abel Sanchez for Group 18. It covers the problem statement, dataset, exploratory analysis, preprocessing, feature correlations, decision-tree methodology, training and testing approach, performance metrics, Actual vs. Predicted Grades visualization, challenges, and conclusion.

## Running the project

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Open the notebook:

```bash
jupyter notebook student_grade_prediction.ipynb
```

Task 2 can also be run separately:

```bash
python task2_preprocessing.py
```
