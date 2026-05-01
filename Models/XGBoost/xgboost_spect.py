import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import xgboost as xgb
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import shap

#####     DATA     #####
# data import
columns = ["Class", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11",
           "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23"] 
training_data = pd.read_csv("SPECT.train", header=None, na_values='?', names=columns)
test_data = pd.read_csv("SPECT.test", header=None, na_values='?', names=columns)

# training data
X_train = training_data.drop(columns=["Class"])
y_train = training_data["Class"]

# test data
X_test = test_data.drop(columns=["Class"])
y_test = test_data["Class"]

### split data based on representative numbers
data = pd.concat([training_data, test_data])
X = data.drop(columns=["Class"])
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
count_negative = (y_train == 0).sum()
count_positive = (y_train == 1).sum()
scale_weight = count_negative / count_positive


#####     MODEL     #####
# set up XGBoost model
est = xgb.XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss',
    scale_pos_weight=scale_weight, # Handles the class imbalance
    random_state=42
)

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=est,
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best Parameters: {grid_search.best_params_}")
model = grid_search.best_estimator_


#####     RESULTS     #####
# plot training confusion matrix
y_pred_train = model.predict(X_train)
cm_train = confusion_matrix(y_train, y_pred_train)
print(cm_train)

# plot test confusion matrix
y_pred_test = model.predict(X_test)
# y_probs = model.predict_proba(X_test)  # Returns the probability for each class
cm_test = confusion_matrix(y_test, y_pred_test)
print(cm_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test,
                              display_labels=model.classes_)
disp.plot()
plt.savefig('confusion_mat_XGBspect.png', bbox_inches='tight')
plt.close()

#calculate accuracy
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {accuracy}")

# calculate F1 score
f1_per_class = f1_score(y_test, y_pred_test, average=None)
print("F1 score per class:", f1_per_class)
f1_macro = f1_score(y_test, y_pred_test, average='macro')
print("Macro-average F1 score:", f1_macro)


#####     SHAP     #####
# setup SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
# shap.initjs()

# # waterfall plot
# shap.waterfall_plot(shap_values[0])

# # force plot
# shap.force_plot(explainer.expected_value, shap_values[0].values, X_test.iloc[0, :], matplotlib=True)

# stacked force plot
# for i in range(100):
    # shap.force_plot(explainer.expected_value, shap_values[i].values, X_test.iloc[i, :], matplotlib=True)

# summary plot
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('XGBspect_shap_summary.png', bbox_inches='tight')
plt.close()

# bar plot of mean SHAP values
# shap.summary_plot(shap_values, X_test, plot_type="bar")
