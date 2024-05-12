import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

def load_data(file_path, label):
    features = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            features.append([float(value.split(':')[1]) for value in parts[1:]])
    labels = [label] * len(features)
    return features, labels

real_features, real_labels = load_data('real.txt', 1)
fake_features, fake_labels = load_data('fake.txt', 0)

features = np.array(real_features + fake_features)
labels = np.array(real_labels + fake_labels)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    model = SGDClassifier(loss='log', max_iter=1, tol=None, learning_rate='constant', eta0=0.01, random_state=42, verbose=0, warm_start=True)
    
    n_epochs = 1000  # 可以调整为所需的epoch数量
    for epoch in range(n_epochs):
        model.fit(X_train, y_train)  # 注意，我们设置warm_start=True来实现连续训练
        preds = model.predict_proba(X_test)
        loss = log_loss(y_test, preds)
        print(f'Epoch {epoch + 1}/{n_epochs}, Log-Loss: {loss}')
