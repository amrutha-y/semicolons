import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Step 1: Read the input CSV file
input_file = "train_cosmic.csv"
data = pd.read_csv(input_file)

# Step 2: Preprocess the data

features = ['Protein Domain','AI Domain','Degree','Betweeness','Eigen Closness','Trails','Drugs',
            'Cosmic','Cancer gene','Publications']

X = data[features]  # Features
y = data['Class']    # Target variable

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: Train the SVR model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svr_model = SVR(kernel='rbf')  # Radial Basis Function kernel
svr_model.fit(X_train, y_train)

# Step 4: Predict scores for genes
predictions = svr_model.predict(X_test)

#step 5 : output data
output_data = {
    'Score': predictions
}
output_df = pd.DataFrame(output_data)

# Step 6: Write predictions to output CSV file
output_df.to_csv("predictions.csv", index=False)
