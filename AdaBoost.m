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

%% 2. Define the AdaBoost Model
% 'AdaBoostM1' is the standard algorithm for binary classification.
% 'NumLearningCycles' is the number of weak learners (trees).
% 'Learners', 'Tree' uses decision stumps by default for boosting.
numTrees = 50; 
t = templateTree('MaxNumSplits', 1); % Decision Stumps work best with AdaBoost

adaModel = fitcensemble(X_train, Y_train, ...
    'Method', 'AdaBoostM1', ...
    'NumLearningCycles', numTrees, ...
    'Learners', t, ...
    'LearnRate', 0.1); % Lower learning rate can prevent overfitting

%% 3. Predict and Evaluate
[predictions, scores] = predict(adaModel, X_test);

% Calculate Accuracy
accuracy = sum(predictions == Y_test) / length(Y_test) * 100;
fprintf('AdaBoost Accuracy: %.2f%%\n', accuracy);

%% 4. Visualization: Confusion Matrix
figure;
confusionchart(Y_test, predictions, ...
    'Title', 'SPECT Heart Diagnosis - AdaBoost Confusion Matrix', ...
    'ColumnSummary', 'column-normalized', ...
    'RowSummary', 'row-normalized');

%% 5. Analyze Error over Iterations
figure;
plot(loss(adaModel, X_test, Y_test, 'Mode', 'cumulative'));
xlabel('Number of Trees');
ylabel('Test Error');
title('Error Convergence of AdaBoost');
grid on;