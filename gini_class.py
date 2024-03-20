import pandas as pd
import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, feature_names=None):
        self.max_depth = max_depth
        self.feature_names = feature_names

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            print(f"At depth {depth}, reached leaf node. Class label: {np.bincount(y).argmax()}")
            return np.bincount(y).argmax()  # Return the most common class label in case of leaf node
        else:
            best_split = self._find_best_split(X, y)
            if best_split is None:
                print(f"At depth {depth}, reached leaf node. Class label: {np.bincount(y).argmax()}")
                return np.bincount(y).argmax()

            feature_index, threshold = best_split
            feature_name = self.feature_names[feature_index]  # Get the feature name
            print(
                f"At depth {depth}, splitting on feature {feature_name} (index: {feature_index}) at threshold {threshold}")

            left_idxs = X[:, feature_index] <= threshold
            right_idxs = ~left_idxs

            left_tree = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
            right_tree = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)

            return {"feature_name": self.feature_names[feature_index],  # Save feature name instead of index
                    "threshold": threshold,
                    "left": left_tree,
                    "right": right_tree}

    def _find_best_split(self, X, y):
        best_gini = float('inf')
        best_split = None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_idxs = X[:, feature_index] <= threshold
                right_idxs = ~left_idxs

                gini_left = self._calculate_gini(y[left_idxs])
                gini_right = self._calculate_gini(y[right_idxs])

                gini = (len(y[left_idxs]) * gini_left + len(y[right_idxs]) * gini_right) / len(y)

                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_index, threshold)

        return best_split

    def _calculate_gini(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def predict(self, X):
        predictions = []
        for sample in X:
            node = self.tree
            while isinstance(node, dict):
                if sample[self.feature_names.index(node["feature_name"])] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]
            predictions.append(node)
        return predictions

    def _print_tree_structure(self, node, depth=0):
        if isinstance(node, dict):
            print("  " * depth,
                f"Split on feature {node['feature_name']}, threshold {node['threshold']}")  # Print feature name
            self._print_tree_structure(node["left"], depth + 1)
            self._print_tree_structure(node["right"], depth + 1)
        else:
            print("  " * depth, f"Leaf node. Class label: {node}")

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
    "Delay": ["yes", "no", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes", "yes", "yes", "yes", "no"]
}
df = pd.DataFrame(data)

# Normalize numerical features
numerical_features = ["Vendor Experience (Years)", "Order Volume (Units)", "Lead Time (Days)",
                      "Number of Previous Delays"]
for feature in numerical_features:
    min_val = df[feature].min()
    max_val = df[feature].max()
    df[feature] = (df[feature] - min_val) / (max_val - min_val)

# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=["Weather Conditions", "Vendor Location"], drop_first=True)

# Split the dataset into features (X) and target variable (y)
X = df_encoded.drop("Delay", axis=1).values
y = df_encoded["Delay"].map({"yes": 1, "no": 0}).values  # Convert "yes" and "no" to 1 and 0

# Store feature names for printing tree structure
feature_names = df_encoded.drop("Delay", axis=1).columns

# Initialize and train the decision tree classifier
tree = DecisionTreeClassifier(max_depth=12, feature_names=feature_names)
tree.fit(X, y)

# Print the tree structure
tree._print_tree_structure(tree.tree)

# Get user input for test data
print("Enter the values for the test data:")

# Prompt for numerical features
test_data = {}
for feature in numerical_features:
    value = float(input(f"Enter the value for {feature}: "))
    min_val = df[feature].min()
    max_val = df[feature].max()
    normalized_value = (value - min_val) / (max_val - min_val)
    test_data[feature] = normalized_value

# Prompt for weather conditions
print("Choose the weather conditions:")
print("1. Sunny\n2. Rainy\n3. Cloudy\n4. Snowy\n5. Stormy\n6. Windy")
weather_option = int(input("Enter the option number: "))
weather_mapping = {1: "Sunny", 2: "Rainy", 3: "Cloudy", 4: "Snowy", 5: "Stormy", 6: "Windy"}
weather_value = weather_mapping.get(weather_option, "Sunny")
for condition in ["Sunny", "Rainy", "Cloudy", "Snowy", "Stormy", "Windy"]:
    test_data[f"Weather Conditions_{condition}"] = 1 if condition == weather_value else 0

# Prompt for vendor location
print("Choose the vendor location:")
print("1. Urban\n2. Suburban\n3. Rural")
location_option = int(input("Enter the option number: "))
location_mapping = {1: "Urban", 2: "Suburban", 3: "Rural"}
location_value = location_mapping.get(location_option, "Urban")
for location in ["Urban", "Suburban", "Rural"]:
    test_data[f"Vendor Location_{location}"] = 1 if location == location_value else 0

# Convert the test data into a DataFrame
X_test = pd.DataFrame([test_data])

# Make predictions
prediction = tree.predict(X_test.values)

# Convert prediction back to "yes" or "no"
prediction = "yes" if prediction[0] == 1 else "no"

# Print prediction
print("Predicted Delay:", prediction)

