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

### Visualizing Loss

Loss can be visualized as vertical arrows from actual data points to the model's prediction line.

<img width="663" height="297" alt="image" src="https://github.com/user-attachments/assets/87045b11-a5bc-4397-8618-8801c62a346e" />

> The red arrows show the **loss** — shorter arrows mean better predictions.

> *Shorter arrows = lower loss = better model*

### MAE vs MSE — How They Handle Outliers

The biggest practical difference between MAE and MSE is how they treat **outliers**.

| Aspect                    | MSE (L2)                                      | MAE (L1)                                      |
|---------------------------|-----------------------------------------------|-----------------------------------------------|
| Penalty on large errors   | Very high (because of squaring)               | Moderate (linear penalty)                     |
| Effect on model           | Pulls the line more toward outliers           | Less affected by outliers                     |
| Best when                 | Outliers are important / valid                | Dataset has many outliers                     |
| Interpretability          | Less intuitive                                | More human-readable (average error)           |

#### Visual Comparison
##### **MSE**: Model gets pulled closer to outliers.
  - **MSE is better** when big mistakes are very costly (e.g., predicting rocket trajectory, medical dosage, stock trading).
<img width="674" height="309" alt="image" src="https://github.com/user-attachments/assets/fe46cf64-564d-4c8e-868f-39ae0a740c52" />
 

##### **MAE**: Model stays closer to the majority of normal data points.
  - **MAE is better** when you have many unusual cases (e.g., predicting prices in a city with both slums and billionaire houses).
<img width="675" height="316" alt="image" src="https://github.com/user-attachments/assets/1ce3bf8f-2ec8-45c7-8fda-7a773ab296bf" />

### When to Choose Which Loss?
#### **Choose MSE / RMSE when:**
- Large errors are **very costly** and you want the model to take them seriously.
- Outliers are **genuine and important** (not noise).
- You are using **Gradient Descent** (smoother optimization due to mathematical properties).

<details>
    <summary>Click to view examples</summary>

- Rocket trajectory prediction — Used in SpaceX and ISRO AI systems. A big error can cause mission failure, so MSE is preferred to heavily penalize large deviations.
- Stock price prediction — Used by trading platforms and Grok/xAI financial analysis tools. Big misses can lead to huge financial losses, hence MSE is commonly used.
- Weather forecasting (temperature) — Used by IMD, Google Weather, and Alibaba Cloud AI. Large errors can affect millions of people, so MSE helps penalize big mistakes strongly.
- Credit risk scoring — Used by banks and fintech companies (like Paytm, PhonePe). Banks want to heavily penalize big mispredictions of default risk.
    
</details>

#### **Choose MAE when:**
- Your dataset has **many outliers** or noisy data that you don’t want to dominate the model.
- You want the loss to be **easily interpretable** (“On average, we are wrong by ₹1.8 lakhs”).
- Robustness is more important than precision.

<details>
    <summary>Click to view examples</summary>

- House price prediction in India — Cities have both cheap houses and ultra-luxury villas (outliers). Many real estate AI tools prefer MAE so the model is not skewed by expensive properties.
- Salary prediction — Few people have extremely high salaries (CEOs, actors, cricketers). MAE keeps predictions stable for normal employees.
- Traffic time prediction — Occasional accidents create outliers. Used by many AI assistants including Perplexity and Grok.
- Medicine price prediction — Some rare drugs are extremely expensive. MAE prevents distortion from these outliers.
    
</details>

#### Use RMSE for final reporting because:
- It gives error in the same unit as the label (e.g., ±2.3 MPG, ±₹45,000, ±3.2°C).
- Very popular in Kaggle competitions and business dashboards.

---

## Gradient Descent in Linear Regression
**Gradient Descent** is a mathematical optimization algorithm used during **training** to iteratively find the **best weights and bias** that minimize the loss function.

Instead of trying every possible combination of weights and biases (which would take forever), gradient descent is smart — it starts somewhere, checks which direction reduces the error, takes a small step in that direction, and repeats until it can no longer improve.

### Core Idea

The model doesn't know the best weights upfront. It starts with random values near zero and **gradually adjusts them** by always asking: *"If I tweak this weight slightly, does the loss go up or down?"* — then moves in the direction that reduces it.

> Think of it like being blindfolded on a hilly landscape, trying to reach the lowest valley. You can't see the whole map, but you can feel the slope under your feet. At every step, you move in the direction that feels most downhill. Eventually, you reach the bottom.

### How Gradient Descent Works

Gradient descent repeats the following four steps for a number of user-defined iterations:

1. **Calculate the loss** — use the current weight and bias to make predictions, then measure how wrong they are using the loss function.
2. **Find the gradient** — compute the direction (and steepness) in which adjusting the weight or bias would reduce loss the most.
3. **Update the parameters** — move the weight and bias a small amount in that direction. The size of this step is controlled by the **learning rate**.
4. **Repeat** — go back to step 1. Continue until further iterations no longer reduce the loss — the model has **converged**.

<img width="508" height="376" alt="image" src="https://github.com/user-attachments/assets/fa51b880-6a49-4618-8033-6fab2d5fd6f8" />

> **Note:** The model begins with randomized weights and biases near zero, not at the correct values. Training is the process of discovering those correct values.

### The Learning Rate

The **learning rate** controls how large each step is during gradient descent. It is one of the most important hyperparameters to tune.

| Learning Rate | Effect |
|---------------|--------|
| **Too large** | Overshoots the minimum — loss bounces around and may never converge |
| **Too small** | Takes tiny steps — training is correct but extremely slow |
| **Just right** | Loss decreases steadily and converges at a good pace |

### Model Convergence and Loss Curves

**Convergence** means the model has found the weights and bias that produce the **lowest achievable loss** — further training iterations no longer meaningfully improve the result.

When training a model, the most common way to monitor this is through a **loss curve** — a graph that plots loss (y-axis) against the number of iterations (x-axis):

#### Three Phases of a Loss Curve

| Phase | What's Happening |
|-------|-----------------|
| **Steep decline (early iterations)** | Weights are far from optimal; every step dramatically improves the model |
| **Gradual reduction (middle)** | Weights are getting closer; improvements are smaller per step |
| **Flattening (convergence)** | Weights have stabilized near the optimal values; further updates change almost nothing |

A model trained on the MPG dataset typically converges around the **1,000th iteration** — loss drops sharply at first, then gradually levels off.

#### Snapshots During Training

<img width="1604" height="874" alt="image" src="https://github.com/user-attachments/assets/1c97e439-be94-4c8a-8a21-d0e7be4e2dbe" />

| Iteration | Model State | Loss |
|-----------|-------------|------|
| ~2nd | Line tilts away from data — poor predictions <img width="436" height="185" alt="image" src="https://github.com/user-attachments/assets/fe277227-4e11-427a-ad19-b6666cd8a31a" /> | Very high |
| ~400th | Line cuts through data but not at optimal angle <img width="429" height="193" alt="image" src="https://github.com/user-attachments/assets/7b14317f-7336-43b8-84a1-082d59fd9e98" /> | Moderate |
| ~1000th | Line fits the data well — model has converged <img width="433" height="196" alt="image" src="https://github.com/user-attachments/assets/592a1550-7dfb-4fb5-b313-1414baf46de6" /> | Lowest achievable |

> **Note:** A loss of exactly 0 is not the goal. It would mean the model perfectly fits every training point — usually a sign of **overfitting**, meaning the model memorized the training data but won't generalize to new data.

---
