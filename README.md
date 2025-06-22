# Attention-LSTM
# Attention-Based LSTM for Stock Price Forecasting

This project demonstrates a hybrid deep learning model that integrates LSTM with an Attention mechanism to predict Google's daily stock closing prices. The model is trained on 30-day windows of multivariate financial data to forecast the next day's price.

Objective:
- Predict the next day's "Close" price using past values of:
  - "Open"
  - "High"
  - "Low"
  - "Close" (target)
  - "Volume"
- Apply a deep learning architecture that includes attention to better focus on relevant time steps.
- Evaluate performance using RMSE and MAE.

Model Overview:
- Input: 30 time steps × 5 features
- Layers:
  - "LSTM(64)" with "return_sequences=True"
  - "Dropout(0.2)"
  - "Attention" applied over LSTM output
  - "Add()" layer to combine attention context and LSTM
  - "LSTM(32)" to extract sequential context
  - "Dense(1)" for final price prediction
- Optimizer: "Adam" with "learning_rate=1e-4"
- Loss: Mean Squared Error

Performance Metrics:
- Baseline Attention LSTM:
  - RMSE: 3.82
  - MAE: 2.83
 
- After Hyperparameter Tuning:
  - RMSE: 4.41
  - MAE: 3.39
     
These metrics are calculated on the test set and represent how well the model approximates actual prices.
While the tuned model slightly underperformed on test RMSE/MAE, it lays the groundwork for further exploration (e.g., stacking, feature engineering, attention scope tuning).

Visualization

The notebook includes a line plot that overlays:
- Actual closing prices
- Predicted prices (from the Attention-LSTM)

This provides a visual inspection of model accuracy over time.

Conclusion

By incorporating an attention mechanism, this model selectively weights important time steps from the input sequence. It shows improved ability to model market behavior compared to traditional LSTM-only approaches.

