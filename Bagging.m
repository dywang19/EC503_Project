%% Bagging
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

%% 2. Define the Base Learner (Decision Tree)
% Bagging works best with deep, unpruned trees to maximize diversity.
t = templateTree('MaxNumSplits', size(X_train, 1) - 1);

%% 3. Train the Bagging Ensemble
% 'Method', 'Bag' creates the bootstrap samples and aggregates them.
numTrees = 100; % Common starting point for bagging
bagModel = fitcensemble(X_train, Y_train, ...
    'Method', 'Bag', ...
    'NumLearningCycles', numTrees, ...
    'Learners', t);

%% 4. Predict and Evaluate
Y_pred = predict(bagModel, X_test);

% Calculate Accuracy
accuracy = sum(Y_pred == Y_test) / length(Y_test) * 100;
fprintf('Bagging Accuracy on SPECT Test Set: %.2f%%\n', accuracy);

%% 5. Visualize Results
figure;
confusionchart(Y_test, Y_pred, 'Title', 'Confusion Matrix: Bagging on SPECT');

% Plot Out-of-Bag (OOB) Error to check convergence
oobError = oobLoss(bagModel, 'Mode', 'cumulative');
figure;
plot(oobError);
xlabel('Number of Trees');
ylabel('Out-of-Bag Error');
title('OOB Error Convergence');
grid on;