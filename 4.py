import pandas as pd

# Create and save data
df = pd.DataFrame({
    "Weather": ["Sunny", "Sunny", "Rainy", "Sunny"],
    "Temperature": ["Warm", "Warm", "Cold", "Warm"],
    "Humidity": ["Normal", "High", "High", "High"],
    "Wind": ["Strong", "Strong", "Strong", "Weak"],
    "PlayTennis": ["Yes", "Yes", "No", "Yes"]
})
df.to_csv("training_data.csv", index=False)
print("CSV 'training_data.csv' created!")

# S-Algorithm
def find_s(data):
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    for i in range(len(y)):
        if y[i] == "Yes":
            hypo = X[i].copy()
            break
    else:
        return "No positive examples found."

    for i in range(len(y)):
        if y[i] == "Yes":
            hypo = ["?" if hypo[j] != X[i][j] else hypo[j] for j in range(len(hypo))]
    return hypo

# Run algorithm
data = pd.read_csv("training_data.csv")
print("\nTraining Data:\n", data)
print("\nFinal Hypothesis:\n", find_s(data))
