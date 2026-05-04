# ML-Concepts

## Linear Regression

**Linear Regression** is a statistical and machine learning technique used to find the relationship between one or more **features** (inputs) and a **label** (output). It models this relationship using a straight line (in simple cases) or a hyperplane (in multiple features).

In ML terms, linear regression learns how to predict a continuous numerical value (like price, temperature, sales, etc.) based on input data.

### Core Idea
As one variable increases or decreases, the target variable tends to change in a predictable linear way. The model draws the "best fit" straight line through the data points that minimizes prediction error.

### Example from Google ML Crash Course
**Problem**: Predict a car's **fuel efficiency** (miles per gallon) based on its **weight**.

**Dataset**:

| Pounds (in 1000s) | Miles per Gallon |
|-------------------|------------------|
| 3.5               | 18               |
| 3.69              | 15               |
| 3.44              | 18               |
| 3.43              | 16               |
| 4.34              | 15               |
| 4.42              | 14               |
| 2.37              | 24               |

**Visualization Insight**:  
Heavier cars generally have lower MPG. When plotted, the points show a downward trend.

**Best Fit Line**:  
A straight line drawn through these points that best represents the overall relationship.

### Linear Regression Equation (Algebra)

In basic mathematics:
```
y = mx + b
```

Where:
- **y** = predicted value (label)
- **m** = slope of the line (how much y changes when x increases by 1)
- **x** = input feature
- **b** = y-intercept (value of y when x = 0)

### Linear Regression in Machine Learning Terms

Google's formulation:

<img width="142" height="44" alt="image" src="https://github.com/user-attachments/assets/03440172-efda-4768-a3eb-7d759617e5a7" />

(For single feature)

Where:
- **y'** = predicted label (what the model outputs)
- **b** = **bias** (same as y-intercept; also sometimes called $w_0$)
- **w₁** = **weight** of the feature (same as slope $m$)
- **x₁** = feature (input)

<img width="684" height="312" alt="image" src="https://github.com/user-attachments/assets/6f1b8a9f-8d44-489c-b9bb-dbb79a795497" />

**Bias (b)**: Tells where the line crosses the y-axis starting point. It shifts the entire line up or down.  
**Weight (w)**: Controls the steepness and direction of the line. Positive weight = upward slope, negative = downward slope.

> During **training**, the model automatically learns the best values of **bias** and **weights** from the data to minimize errors.
<img width="681" height="268" alt="Screenshot 2026-05-04 at 11 16 11 AM" src="https://github.com/user-attachments/assets/cd97d7bd-4dbc-4d0d-98b2-a1762c8a92ab" />

**Example Calculation** (from the course):  
Bias ≈ 34, Weight ≈ -4.6  
Model: `MPG = 34 + (-4.6) × (weight in 1000s)`

Prediction for a 4000-pound car:  
`MPG = 34 - 4.6 × 4 = 34 - 18.4 = 15.6`

### Models with Multiple Features

Real-world problems usually have many features. The equation becomes:

<img width="395" height="42" alt="image" src="https://github.com/user-attachments/assets/ba39b683-0235-4637-856b-342d7faa8a7b" />

**Example** — Predicting MPG using multiple features:
- Weight
- Engine displacement
- Acceleration (0-60 time)
- Number of cylinders
- Horsepower

The model learns a separate **weight** for each feature.

<img width="672" height="253" alt="image" src="https://github.com/user-attachments/assets/50cabf35-e866-41da-b2b8-54df8ac7a4e5" />


**Key Observations** (from course):
- Bigger engine displacement → generally lower MPG (negative weight)
<img width="1352" height="802" alt="image" src="https://github.com/user-attachments/assets/e1b676a7-43fe-4876-bd01-95b7c70c3560" />

- Slower acceleration (higher 0-60 time) → can show positive relationship in some cases
<img width="1374" height="798" alt="image" src="https://github.com/user-attachments/assets/1d095457-6c48-4cf1-a681-33746ca7e3c7" />


### Important Terms

| Term              | Meaning                                                                 | Analogy                          |
|-------------------|-------------------------------------------------------------------------|----------------------------------|
| **Feature**       | Input variable used for prediction                                      | Ingredients in cooking           |
| **Label**         | Output value we want to predict                                         | Final dish                       |
| **Bias (b)**      | Base value when all features are zero                                   | Starting point                   |
| **Weight (w)**    | How much each feature influences the prediction                         | Importance / multiplier of each ingredient |
| **Training**      | Process of finding optimal bias & weights                               | Learning from past recipes       |
| **Prediction**    | Using the learned equation on new data                                  | Cooking a new dish               |

<details>
    <summary>Click to view Simple Real-Life Analogies</summary>

### Simple Real-Life Analogies

1. **House Price Prediction**:
   - Features: Size (sqft), Bedrooms, Location score
   - Label: Price in rupees
   - Model learns: "Every extra sqft adds ~₹5000, each bedroom adds ~₹8 lakhs, etc."

2. **Exam Score Prediction**:
   - Features: Hours studied, Sleep hours, Previous test scores
   - Label: Final exam marks
   - Linear regression finds how strongly each factor affects the score.

3. **Cooking Analogy**:
   - Salt, Sugar, Spice, Cooking time → Features
   - Final taste rating → Label
   - Model learns the "recipe coefficients" (weights)

### When to Use Linear Regression

**Good for**:
- Predicting continuous values (prices, temperatures, sales, age, etc.)
- When relationship looks roughly linear
- As a baseline model (simple & interpretable)

**Limitations**:
- Assumes linear relationship (may fail if relationship is curved)
- Sensitive to outliers
- Can struggle with very complex real-world patterns (that's where advanced models come in)

</details>

#### Questions:
- What parts of the linear regression equation are updated during training?
     - The bias and weights

---

## Loss Function in Linear Regression

**Loss** is a numerical metric that tells **how wrong** a model's predictions are.  

The main goal during training is to **minimize the loss** — i.e., make the model's predictions as close as possible to the actual labels.

### What is Loss?

Loss measures the **distance** between the model's **predicted value** and the **actual label**.
> In machine learning, **Loss = Distance between Predicted Value and Actual Value**.

- It focuses on **distance**, not direction.
- If the model predicts **2** but actual value is **5**, the difference is **3** (we ignore the negative sign).
    - If model predicts **₹8 lakhs** but actual house price is **₹10 lakhs**, loss = 2 lakhs (we ignore negative).

**Two common ways to remove the sign**:
1. Take the **absolute value** → L1 Loss
2. **Square** the difference → L2 Loss

<details>
    <summary>Click to view Real-Life Scenarios to Understand Loss</summary>

### Real-Life Scenarios to Understand Loss

1. **House Price Prediction** (Most Common Example)
   - Feature: House size (sq ft)
   - Label: Actual selling price
   - Model predicts ₹45 lakhs for a house, but it actually sold for ₹52 lakhs.
   - **Loss** tells the agent how wrong the prediction was.

2. **Fuel Efficiency Prediction** (Google Course Example)
   - Feature: Car weight
   - Label: Actual Miles Per Gallon (MPG)
   - Model predicts 23.1 MPG, actual = 24 MPG → Small loss.

3. **Exam Score Prediction**
   - Features: Hours studied, attendance, sleep hours
   - Label: Actual marks (out of 100)
   - Model predicts 78, student scored 65 → High loss for this student.

4. **Weather Forecasting**
   - Feature: Humidity, wind speed, pressure
   - Label: Actual temperature
   - Model predicts 32°C, actual = 29°C → Loss of 3 degrees.
    
</details>

### Types of Loss Functions
- In linear regression, there are five main types of loss, which are outlined in the following table.

<img width="732" height="432" alt="image" src="https://github.com/user-attachments/assets/f4c91fef-751f-46d2-9d3d-8e9835e970e7" />

> The functional difference between L1 loss and L2 loss (or between MAE/RMSE and MSE) is squaring. When the difference between the prediction and label is large, squaring makes the loss even larger. When the difference is small (less than 1), squaring makes the loss even smaller.
> Loss metrics like MAE and RMSE may be preferable to L2 loss or MSE in some use cases because they tend to be more human-interpretable, as they measure error using the same scale as the model's predicted value.

- **Note:** _MAE and RMSE can differ quite widely. MAE represents the average prediction error, whereas RMSE represents the "spread" of the errors, and is more skewed by larger errors._
- _When processing multiple examples at once, we recommend averaging the losses across all the examples, whether using MAE, MSE, or RMSE._

### Loss Calculation Example

**Model** :  
<img width="722" height="661" alt="image" src="https://github.com/user-attachments/assets/d63ad6b0-338c-4c65-a1a0-16bb7364d7d2" />

---

### Visualizing Loss

Loss can be visualized as vertical arrows from actual data points to the model's prediction line.

![Loss Visualization](images/loss_arrows.png)

