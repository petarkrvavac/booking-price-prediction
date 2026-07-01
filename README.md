# Hotel Booking Price Prediction

A machine learning project focused on predicting hotel booking prices using the **Hotel Booking Demand** dataset.
The goal of the project is to estimate the **Average Daily Rate (ADR)** of a hotel reservation based on booking-related features such as hotel type, arrival date, number of guests, room type, market segment and length of stay.

The project includes data cleaning, exploratory data analysis, feature engineering, model training and performance comparison between a baseline regression model and a more advanced ensemble model.

---

## Overview

Hotel pricing depends on many factors, including seasonality, demand, room type, booking channel and guest information.
This project treats hotel price prediction as a **regression problem**, where the target variable is:

```text
adr
```

ADR stands for **Average Daily Rate** and represents the average daily price of a hotel booking.

Two machine learning models were trained and evaluated:

* Linear Regression
* Random Forest Regressor

The Random Forest model achieved significantly better results and was able to capture more complex pricing patterns in the dataset.

---

## Dataset

The project uses the **Hotel Booking Demand** dataset, which contains reservation records from two types of hotels:

* City Hotel
* Resort Hotel

The dataset includes information about:

* arrival date
* length of stay
* number of guests
* hotel type
* meal type
* room type
* market segment
* distribution channel
* customer type
* booking changes
* special requests
* average daily rate

Initial dataset size:

```text
119,390 rows
32 columns
```

---

## Data Preprocessing

Before model training, the dataset was cleaned and prepared through several steps:

* removed duplicate rows
* handled missing values
* removed irrelevant columns with too many missing values
* removed invalid ADR values
* handled extreme ADR outliers using the IQR method
* removed bookings with zero guests
* converted categorical variables using one-hot encoding
* created additional features for better model performance

Columns that could cause data leakage were also removed before training, such as cancellation status and reservation outcome information.

---

## Feature Engineering

Several new features were created from the original dataset:

```text
total_guests = adults + children + babies
```

```text
total_nights = stays_in_weekend_nights + stays_in_week_nights
```

The arrival month was also converted into a numerical feature to help the model capture seasonality.

Countries with fewer occurrences were grouped into a common `"Other"` category to reduce dimensionality and improve model stability.

---

## Models

### Linear Regression

Linear Regression was used as a baseline model.
It provides a simple and interpretable starting point, but it has limited ability to model nonlinear relationships in hotel pricing.

### Random Forest Regressor

Random Forest Regressor was used as the main predictive model.
It performs better on this dataset because it can capture nonlinear relationships and interactions between features such as season, hotel type, room type and market segment.

Main parameters:

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
```

---

## Results

Model performance was evaluated using MAE, RMSE and R² score.

| Model                   |       MAE |      RMSE | R² Score |
| ----------------------- | --------: | --------: | -------: |
| Linear Regression       | 24.29 EUR | 32.31 EUR |   0.4396 |
| Random Forest Regressor | 11.51 EUR | 17.18 EUR |   0.8416 |

The Random Forest model achieved the best performance, with an average prediction error of approximately **11.51 EUR** and an **R² score of 0.8416**.

This means that the model explains around **84% of the variance** in hotel booking prices.

---

## Key Insights

The results show that hotel booking prices are strongly influenced by:

* seasonality
* number of guests
* hotel type
* room type
* market segment
* booking lead time
* meal type
* special requests

The comparison between models also shows that hotel pricing is not purely linear.
A tree-based ensemble model performed much better because it was able to capture more complex relationships in the data.

---

## Technologies Used

* Python
* Jupyter Notebook
* pandas
* NumPy
* Matplotlib
* Seaborn
* scikit-learn

---

## Project Structure

```text
hotel-booking-price-prediction/
│
├── booking-price-prediction.ipynb
├── hotel_bookings.csv
└── README.md
```

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/your-username/hotel-booking-price-prediction.git
```

Open the project folder:

```bash
cd hotel-booking-price-prediction
```

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Run the Jupyter Notebook:

```bash
jupyter notebook booking-price-prediction.ipynb
```

Make sure the dataset file is placed in the same directory as the notebook.

---

## Future Improvements

Possible improvements for future versions:

* test XGBoost, LightGBM or CatBoost
* perform hyperparameter tuning
* use cross-validation
* add external data such as holidays, local events or weather
* build a simple web interface for price prediction
* deploy the trained model as an API

---

## Conclusion

This project demonstrates how machine learning can be used to predict hotel booking prices from reservation data.

The Random Forest Regressor achieved the best results and showed that hotel prices depend on complex interactions between booking attributes. With further tuning and additional external data, this type of model could be extended into a practical dynamic pricing or hotel revenue management tool.
