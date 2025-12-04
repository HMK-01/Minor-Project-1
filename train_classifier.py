import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Take 10% as final test set
x_temp, x_test, y_temp, y_test = train_test_split(
    data,
    labels,
    test_size=0.10,      # 10% test
    shuffle=True,
    stratify=labels,
    random_state=42
)

# Split remaining 90% into 75% train and 15% val overall
# 15 / (75 + 15) = 0.166666...
x_train, x_val, y_train, y_val = train_test_split(
    x_temp,
    y_temp,
    test_size=0.1666667,  # ~16.67% of 90% -> 15% overall
    shuffle=True,
    stratify=y_temp,
    random_state=42
)

# Train model on training set
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Model Accuracy
y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)
y_test_pred = model.predict(x_test)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Train accuracy: {train_acc * 100:.2f}%")
print(f"Validation accuracy: {val_acc * 100:.2f}%")
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Save model trained on 75% train data
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
