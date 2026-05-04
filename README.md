# ML-Concepts
## Linear Regression
Linear regression is a statistical technique used to find the relationship between variables. In an ML context, linear regression finds the relationship between features and a label.
- In machine learning (including linear regression), the terms **feature** and **label** are just simple ways of describing *inputs* and *outputs*.

### Feature (Input)

A **feature** is any piece of information you use to make a prediction.

Think of features as the **clues** or **independent variables**.

* If you're predicting house prices:

  * Size of the house (sq ft)
  * Number of bedrooms
  * Location
  > These are all **features**

You can have:

* One feature → simple linear regression
* Multiple features → multiple linear regression

### Label (Output)

A **label** is what you're trying to predict.

Think of it as the **answer** or **dependent variable**.

* In the same house example:

  * House price
  > This is the **label**

### Simple Analogy

Imagine you're trying to guess a student's exam score:

* Hours studied → feature
* Sleep hours → feature
* Exam score → label

> Features go **in**, label comes **out**

### How Linear Regression Uses Them

Linear regression tries to find a formula like:

```
Label = w₁ × Feature₁ + w₂ × Feature₂ + ... + b
```

Example:

```
House Price = (price per sq ft × size) + constant
```

It learns the weights (w₁, w₂, etc.) from data.

### Real-Life Intuition

Think of it like cooking:

* Ingredients → features
* Final dish → label

The model learns *how much of each ingredient* affects the final result.

---

