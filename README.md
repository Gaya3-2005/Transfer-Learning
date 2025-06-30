# Transfer-Learning
# Transfer Learning on Tabular Data: Customer Segmentation to Store Dataset

This project explores the application of **transfer learning** for **tabular data** using FastAI. The goal is to train a model on a customer segmentation dataset and fine-tune it on an online retail dataset to observe performance, adaptability, and clustering behavior.


## Datasets Used

- **Model 1 Dataset (Customer Segmentation)**  
  Features include `Customer_Age`, `Gender`, `Product_Category`, `Revenue`, `Profit`, etc.

- **Model 2 Dataset (Store Dataset)**  
  Includes `InvoiceNo`, `StockCode`, `UnitPrice`, `Quantity`, `Revenue`, and `CustomerID`.

---

## Tools & Technologies

- Python 3.11  
- FastAI (Tabular)  
- Scikit-learn  
- Matplotlib, Seaborn  
- Google Colab  

---

## Workflow

### Step 1: Train on Model 1 Dataset
- Preprocessed using `Categorify`, `FillMissing`, and `Normalize`
- Target: `Product_Category`
- Saved model: `product_category_model.pth`
- Achieved 100% validation accuracy

### Step 2: Transfer Learning on Model 2 Dataset
- Loaded saved model and replaced dataloaders
- Fine-tuned on new dataset (different features and distribution)
- Accuracy dropped to ~47% → highlighting domain shift

---

## Evaluation

- **Before Training Accuracy**: 1.0000  
- **After Transfer Accuracy**: 0.4700  
- **Metrics Used**:
  - Accuracy
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1)
- Visualizations:
  - Clustering (K-Means)
  - PCA dimensionality reduction

---

## Clustering Comparison

- **Before Training**: Clustered using raw `Revenue` and `Profit`
- **After Training**: Clustered again using same variables
- Observed that transfer learning did not significantly change cluster boundaries, indicating limited generalization.

---

## Inference

```python
row = df2.iloc[0]
learn.predict(row)
