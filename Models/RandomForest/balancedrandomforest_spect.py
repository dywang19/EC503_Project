import pandas as pd
import sklearn
import imblearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
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

### split data based on UCI database
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


#####     MODEL     #####
# set up model
est = imblearn.ensemble.BalancedRandomForestClassifier(
    class_weight='balanced_subsample', random_state=42)

# parameters to CV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

# set up grid search CV using F1 macro-avg
grid_search = GridSearchCV(
    estimator=est, 
    param_grid=param_grid, 
    cv=5,              # 5-fold cross-validation
    scoring='f1_macro',      # Focus on balancing precision and recall
    n_jobs=-1,         # Use all available CPU cores
    verbose=1
)
# find and use best model
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
plt.savefig('confusion_mat_BRFspect.png', bbox_inches='tight')
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
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# waterfall plot
shap.plots.waterfall(shap_values[0, :, 0], show=False)
plt.savefig('BRFspect_waterfall.png', bbox_inches='tight')
plt.close()

# force plot
# shap.force_plot(explainer.expected_value, shap_values[0].values, X_test.iloc[0, :], matplotlib=True)

# stacked force plot
# for i in range(100):
    # shap.force_plot(explainer.expected_value, shap_values[i].values, X_test.iloc[i, :], matplotlib=True)

# summary plot
# shap.summary_plot(shap_values, X_test, show=False)
shap.plots.beeswarm(shap_values[:, :, 1], show=False)
plt.savefig('BRFspect_shap_summary.png', bbox_inches='tight')
plt.close()

# bar plot of mean SHAP values
# shap.summary_plot(shap_values, X_test, plot_type="bar")
