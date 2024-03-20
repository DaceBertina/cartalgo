import pandas as pd
import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None, feature_names=None, categorical_features=None):
        self.max_depth = max_depth
        self.feature_names = feature_names
        self.categorical_features = categorical_features

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
        print("Decision Tree Structure:")
        self._print_tree_structure(self.tree)

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(set(y)) == 1:
            print(f"At depth {depth}, reached leaf node. Mean of Delay Duration (Days): {np.mean(y):.3f}")
            return np.mean(y)

        best_split = self._find_best_split_variance(X, y)
        if best_split is None:
            print(f"At depth {depth}, reached leaf node. Mean of Delay Duration (Days): {np.mean(y):.3f}")
            return np.mean(y)

        feature_index, threshold, variance_reduction = best_split
        feature_name = self.feature_names[feature_index]  # Get the feature name
        print(
            f"At depth {depth}, splitting on feature {feature_name} (index: {feature_index}) at threshold {threshold} with variance reduction {variance_reduction:.3f}")

        left_idxs = X[:, feature_index] <= threshold
        right_idxs = ~left_idxs

        left_tree = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_tree = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)

        return {"feature_index": feature_index,
                "feature_name": feature_name,
                "threshold": threshold,
                "variance_reduction": variance_reduction,
                "left": left_tree,
                "right": right_tree}

    def _find_best_split_variance(self, X, y):
        best_variance_reduction = float('-inf')
        best_split = None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_idxs = X[:, feature_index] <= threshold
                right_idxs = ~left_idxs

                var_left = self._calculate_variance(y[left_idxs])
                var_right = self._calculate_variance(y[right_idxs])

                total_variance = self._calculate_variance(y)
                weighted_variance = (len(y[left_idxs]) * var_left + len(y[right_idxs]) * var_right) / len(y)
                variance_reduction = total_variance - weighted_variance

                if variance_reduction > best_variance_reduction:
                    best_variance_reduction = variance_reduction
                    best_split = (feature_index, threshold, variance_reduction)

        return best_split

    def _calculate_variance(self, arr):
        n = len(arr)
        if n == 0:
            return 0
        mean = np.mean(arr)
        return np.sum((arr - mean) ** 2) / n

    def predict(self, X, y_true=None):
        predictions = []
        squared_errors = []
        for sample, true_value in zip(X, y_true):
            node = self.tree
            while isinstance(node, dict):
                # Check if the feature at the split node is categorical
                if node["feature_name"] in self.categorical_features:
                    # One-hot encode the categorical feature
                    feature_value = sample[self.feature_names.index(node["feature_name"])]
                    encoded_feature = [1 if feature_value == value else 0 for value in self.feature_names]
                    # Determine which branch to traverse for categorical feature
                    if encoded_feature[self.feature_names.index(node["threshold"])] == 1:
                        node = node["left"]
                    else:
                        node = node["right"]
                else:
                    # If it's numerical, perform the usual comparison
                    if sample[node["feature_index"]] <= node["threshold"]:
                        node = node["left"]
                    else:
                        node = node["right"]
            # If the current node is a leaf node, extract and use the mean value
            if not isinstance(node, dict):
                predicted_value = node
            else:
                # If the current node is not a leaf node, something went wrong
                raise ValueError("Non-leaf node encountered during prediction.")
            predictions.append(predicted_value)
            squared_errors.append((predicted_value - true_value) ** 2)  # Use true_value for calculating squared error
        mse = np.mean(squared_errors)  # Calculate MSE

        # Ensure predictions list contains only one element before rounding
        if len(predictions) != 1:
            raise ValueError("Multiple or zero predictions found. Unable to determine a single predicted value.")

        # Extract the single prediction from the list before rounding
        single_prediction = predictions[0]

        # Round predicted delay duration and convert to a single value
        predicted_delay = round(
            single_prediction[0] if isinstance(single_prediction, np.ndarray) else single_prediction)

        return predicted_delay, round(mse[0] if isinstance(mse, np.ndarray) else mse, 2)  # Round MSE to two decimal places

    def _print_tree_structure(self, node, depth=0):
        if isinstance(node, dict):
            print("  " * depth,
                  f"Split on {node['feature_name']} (index: {node['feature_index']}), threshold {node['threshold']}")
            self._print_tree_structure(node["left"], depth + 1)
            self._print_tree_structure(node["right"], depth + 1)
        else:
            print("  " * depth, f"Leaf node. Mean of Delay Duration (Days): {node:.3f}")

# Load the dataset
data = {
    "Vendor Experience (Years)": [5, 3, 8, 2, 6, 4, 5, 5, 3, 2, 4, 5, 1, 8],
    "Order Volume (Units)": [100, 80, 200, 50, 150, 120, 100, 100, 50, 100, 150, 200, 100, 100],
    "Lead Time (Days)": [10, 15, 8, 20, 12, 18, 10, 10, 8, 10, 12, 15, 10, 12],
    "Number of Previous Delays": [2, 1, 0, 3, 2, 1, 2, 2, 1, 3, 3, 4, 2, 0],
    "Weather Conditions": ["Sunny", "Rainy", "Snowy", "Rainy", "Cloudy", "Windy", "Snowy", "Stormy", "Windy", "Stormy",
                           "Snowy", "Rainy", "Cloudy", "Snowy"],
    "Vendor Location": ["Urban", "Suburban", "Rural", "Urban", "Rural", "Suburban", "Urban", "Rural", "Rural", "Rural",
                        "Suburban", "Rural", "Rural", "Urban"],
    "Delay Duration (Days)": [3, 2, 0, 5, 4, 3, 6, 1, 2, 5, 4, 3, 4, 0]
    }
df = pd.DataFrame(data)

# Normalize numerical features in the dataset
numerical_features = ["Vendor Experience (Years)", "Order Volume (Units)", "Lead Time (Days)", "Number of Previous Delays", "Delay Duration (Days)"]
for feature in numerical_features:
    min_val = df[feature].min()
    max_val = df[feature].max()
    df[feature] = (df[feature] - min_val) / (max_val - min_val)

# Split the dataset into features (X) and target variable (y)
X = df.drop("Delay Duration (Days)", axis=1).values
y = df["Delay Duration (Days)"].values

feature_names = list(df.columns)
feature_names.remove("Delay Duration (Days)")  # Assuming this is your target variable

# Define the list of categorical features
categorical_features = ["Weather Conditions", "Vendor Location"]

# Initialize and train the decision tree model with increased depth
tree = DecisionTree(max_depth=8, feature_names=feature_names, categorical_features=categorical_features)
tree.fit(X, y)

# Get user input for test data
print("Enter the values for the test data:")

# Prompt for numerical features
numerical_features = ["Vendor Experience (Years)", "Order Volume (Units)", "Lead Time (Days)", "Number of Previous Delays"]
test_data = {}
# Collect user input for numerical features and store them in the test_data dictionary
for feature in numerical_features:
    value = float(input(f"Enter the value for {feature}: "))

    # Check if the user input falls within the range of the training data for this feature
    min_val = min(data[feature])
    max_val = max(data[feature])
    if value < min_val or value > max_val:
        print(f"Warning: The input value for {feature} is outside the range observed in the training data.")

    test_data[feature] = value

# Normalize numerical values and print test data
for feature in numerical_features:
    min_val = min(data[feature])
    max_val = max(data[feature])

    # Normalize only if the user input is within the range of the training data
    if min_val <= test_data[feature] <= max_val:
        normalized_value = (test_data[feature] - min_val) / (max_val - min_val)
        test_data[feature] = normalized_value
    else:
        # If the input value is outside the range, keep it unchanged
        print(
            f"Warning: Skipping normalization for {feature} due to input value outside the range observed in the training data.")

# Prompt for weather conditions
print("Choose the weather conditions:")
print("1. Sunny\n2. Rainy\n3. Cloudy\n4. Snowy\n5. Stormy\n6. Windy")
weather_option = int(input("Enter the option number: "))
weather_mapping = {1: "Sunny", 2: "Rainy", 3: "Cloudy", 4: "Snowy", 5: "Stormy", 6: "Windy"}
weather_value = weather_mapping.get(weather_option, "Sunny")
for condition in ["Sunny", "Rainy", "Cloudy", "Snowy", "Stormy", "Windy"]:
    test_data[f"Weather Conditions_{condition}"] = [1 if condition == weather_value else 0]

# Prompt for vendor location
print("Choose the vendor location:")
print("1. Urban\n2. Suburban\n3. Rural")
location_option = int(input("Enter the option number: "))
location_mapping = {1: "Urban", 2: "Suburban", 3: "Rural"}
location_value = location_mapping.get(location_option, "Urban")
for location in ["Urban", "Suburban", "Rural"]:
    test_data[f"Vendor Location_{location}"] = [1 if location == location_value else 0]

# Convert the test data into a DataFrame
X_test = pd.DataFrame(test_data)

# After obtaining the predictions make predictions and calculate MSE
y_pred_normalized, mse = tree.predict(X_test.values, y)

# Convert y_pred_normalized to a NumPy array
y_pred_normalized = np.array(y_pred_normalized)

# Calculate min and max delay durations before normalization
min_delay_duration_orig = np.min(data["Delay Duration (Days)"])
max_delay_duration_orig = np.max(data["Delay Duration (Days)"])

# Reverse normalization of predicted values using original min and max values
y_pred_orig = y_pred_normalized * (max_delay_duration_orig - min_delay_duration_orig) + min_delay_duration_orig

# Print the predicted delay duration and MSE
print("Predicted Delay Duration (Days):", y_pred_orig)
print("Mean Squared Error (MSE):", mse)

