%% AdaBoost
% load a dataset
%% SPECT Heart
clear; clc;
trainData = readmatrix('SPECT.train', 'FileType', 'text');
testData = readmatrix('SPECT.test', 'FileType', 'text');

% The first column is the label (0 = Normal, 1 = Abnormal)
% The remaining columns are the 22 binary features
X_train = trainData(:, 2:end);
Y_train = trainData(:, 1);
X_test = testData(:, 2:end);
Y_test = testData(:, 1);

%% Breast Cancer Wisconsin
clear; clc;
Data = readmatrix('breast-cancer-wisconsin.data', 'FileType', 'text');
y = Data(:,end);
X = Data(:,2:end-1);

% split data 80/20 train/test, keep class ratio
cv = cvpartition(y, 'Holdout', 0.2);
idxTrain = training(cv);
idxTest = test(cv);

% training data - 80%
X_train = X(idxTrain, :);
Y_train = y(idxTrain);
% test data - 20%
X_test = X(idxTest, :);
Y_test = y(idxTest);

%% Run
% model setup
numTrees = 50; 
t = templateTree('MaxNumSplits', 1);

% AdaBoostM1 for binary classification, decision stumps
adaModel = fitcensemble(X_train, Y_train, 'Method', 'AdaBoostM1', 'NumLearningCycles', numTrees,'Learners', t,'LearnRate', 0.1);

% training predictions
Y_train_pred = predict(adaModel, X_train);
% training accuracy
trainAccuracy = sum(Y_train_pred == Y_train) / length(Y_train) * 100;
fprintf('AdaBoost Training Accuracy: %.2f%%\n', trainAccuracy);

% test predictions
[predictions, scores] = predict(adaModel, X_test);
% test accuracy
accuracy = sum(predictions == Y_test) / length(Y_test) * 100;
fprintf('AdaBoost Accuracy: %.2f%%\n', accuracy);

% confusion matrix
figure;
confusionchart(Y_test, predictions);
title('Breast Cancer Wisconsin AdaBoost Confusion Matrix');
xlabel('Predicted Labels');
ylabel('True Labels');

% 
figure;
plot(loss(adaModel, X_test, Y_test, 'Mode', 'cumulative'));
xlabel('Number of Trees');
ylabel('Test Error');
title('Error Convergence of AdaBoost');
grid on;