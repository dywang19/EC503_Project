import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import shap

#####     DATA     #####
# data import
file_path = "breast-cancer-wisconsin.data"
columns = ["SampleID", "ClumpThickness", "UniformityCellSize",
           "UniformityCellShape", "MarginalAdhesion", "SingleEpithelialCellSize", "BareNuclei",
           "BlandChromatin", "NormalNucleoli", "Mitoses", "Class"]
bcw = pd.read_csv(file_path, header=None, na_values='?', names=columns)

# separate features from classes
bcw['Class'] = bcw['Class'].replace({2: 0, 4: 1})  # 2 is benign, 4 is malignant
bcw = bcw.dropna()  # drop na values
X = bcw.drop(columns=["Class", "SampleID"])
y = bcw["Class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#####     MODEL     #####
# set up AdaBoost model
model = sklearn.ensemble.RandomForestClassifier()
model.fit(X_train, y_train)  # train model


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

# calculate F1 score
f1_per_class = f1_score(y_test, y_pred_test, average=None)
print("F1 score per class:", f1_per_class)
f1_micro = f1_score(y_test, y_pred_test, average='micro')
print("Micro-average F1 score:", f1_micro)
f1_macro = f1_score(y_test, y_pred_test, average='macro')
print("Macro-average F1 score:", f1_macro)
f1_weighted = f1_score(y_test, y_pred_test, average='weighted')
print("Weighted-average F1 score:", f1_weighted)


#####     SHAP     #####
# setup SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
shap.initjs()

# waterfall plot
shap.waterfall_plot(shap_values[0, 0])

# force plot
shap.force_plot(explainer.expected_value, shap_values[0].values, X_test.iloc[0, :], matplotlib=True)

# stacked force plot
# for i in range(100):
    # shap.force_plot(explainer.expected_value, shap_values[i].values, X_test.iloc[i, :], matplotlib=True)

# summary plot
shap.summary_plot(shap_values, X_test)

# bar plot of mean SHAP values
shap.summary_plot(shap_values, X_test, plot_type="bar")
