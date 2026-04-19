import matplotlib.pyplot as plt

# XGBoost data
xgb_coverage = [1.00, 0.78, 0.57, 0.39, 0.25, 0.15, 0.08, 0.03, 0.01]
xgb_accuracy = [0.53, 0.54, 0.55, 0.56, 0.58, 0.59, 0.58, 0.58, 0.62]

# LSTM data
lstm_coverage = [1.00, 0.78, 0.57, 0.38, 0.22, 0.11, 0.05, 0.02, 0.01]
lstm_accuracy = [0.53, 0.54, 0.54, 0.55, 0.56, 0.59, 0.59, 0.61, 0.63]

# Selected threshold points (approx)
xgb_selected = (0.15, 0.59)
lstm_selected = (0.22, 0.56)

# Plot
plt.figure()

plt.plot(xgb_coverage, xgb_accuracy, marker='o', label='XGBoost')
plt.plot(lstm_coverage, lstm_accuracy, marker='o', label='LSTM')

# Highlight selected points
plt.scatter(*xgb_selected, s=100)
plt.scatter(*lstm_selected, s=100)

# Labels and title
plt.xlabel('Coverage')
plt.ylabel('Accuracy')
plt.title('Coverage vs Accuracy Trade-off')
plt.legend()

plt.grid()
plt.show()