% Main script to run Grey Wolf Optimization for ANN
clear; close all; clc;

%% Load and preprocess data
disp('Loading and preprocessing data...');
[X, y, X_norm, feature_names] = preprocessBankData('bank.csv');

% Split data into training and testing sets (Octave compatible)
% Randomly shuffle the data
rand_idx = randperm(size(X_norm, 1));
X_norm = X_norm(rand_idx, :);
y = y(rand_idx);

% Split into 70% training, 30% testing
train_size = floor(0.7 * size(X_norm, 1));
X_train = X_norm(1:train_size, :);
y_train = y(1:train_size);
X_test = X_norm(train_size+1:end, :);
y_test = y(train_size+1:end);

%% Neural Network Parameters
input_layer_size = size(X_train, 2);  % Number of features after preprocessing
hidden_layer_size = 25;               % Hidden layer neurons (can be tuned)
num_labels = length(unique(y_train)); % Number of classes in output layer

%% Grey Wolf Optimization Parameters
max_iter = 50;         % Maximum number of iterations
search_agents_no = 10; % Number of grey wolves (search agents)
lb = -1;               % Lower bound of parameters space
ub = 1;                % Upper bound of parameters space

% Calculate dimension of the problem (total number of weights in the network)
dim = hidden_layer_size * (input_layer_size + 1) + num_labels * (hidden_layer_size + 1);

fprintf('Running Grey Wolf Optimization with %d iterations and %d search agents...\n', max_iter, search_agents_no);
fprintf('Neural Network structure: %d input -> %d hidden -> %d output neurons\n', input_layer_size, hidden_layer_size, num_labels);

% Run Grey Wolf Optimization
best_weights = greyWolfOptimizer(X_train, y_train, input_layer_size, hidden_layer_size, num_labels, max_iter, search_agents_no, lb, ub, dim);

% Reshape the best weights to Theta1 and Theta2
Theta1 = reshape(best_weights(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(best_weights((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%% Evaluate model performance
disp('Evaluating model performance...');

% Calculate training set accuracy
pred_train = predict(Theta1, Theta2, X_train);
train_accuracy = mean(double(pred_train == y_train)) * 100;
fprintf('Training Set Accuracy: %f%%\n', train_accuracy);

% Calculate test set accuracy
pred_test = predict(Theta1, Theta2, X_test);
test_accuracy = mean(double(pred_test == y_train)) * 100;
fprintf('Test Set Accuracy: %f%%\n', test_accuracy);

% Calculate additional metrics for binary classification
if num_labels == 2
    % Confusion matrix
    C = confusionmat(y_test, pred_test);

    % True positives, false positives, true negatives, false negatives
    TP = C(2,2); FP = C(1,2); TN = C(1,1); FN = C(2,1);

    % Precision, recall, F1 score
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1_score = 2 * precision * recall / (precision + recall);

    fprintf('Precision: %f\n', precision);
    fprintf('Recall: %f\n', recall);
    fprintf('F1 Score: %f\n', f1_score);
end

% Plot the convergence curve
figure;
plot(1:max_iter, convergence_curve, 'LineWidth', 2);
title('Convergence Curve');
xlabel('Iteration');
ylabel('Cost Function Value');
grid on;

disp('Optimization completed successfully!');
