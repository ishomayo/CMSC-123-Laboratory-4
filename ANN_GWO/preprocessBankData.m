function [X, y, X_norm, feature_names] = preprocessBankData(filename)
% Preprocess bank.csv dataset for neural network training
% Octave-compatible version
%
% Inputs:
%   filename - Path to the bank.csv file
%
% Outputs:
%   X - Original feature matrix
%   y - Target variable (1/2 for binary classification)
%   X_norm - Normalized feature matrix
%   feature_names - Names of the features after preprocessing

% Read the raw CSV file manually since Octave doesn't have readtable
raw_data = dlmread(filename, ';', 1, 0);
% Load the header separately to get feature names
fid = fopen(filename, 'r');
header_line = fgetl(fid);
fclose(fid);
headers = strsplit(header_line, ';');

% We need to load the text data separately since dlmread only gets numeric data
% Using textread to get categorical columns
[age, job, marital, education, default, balance, housing, loan, ...
 contact, day, month, duration, campaign, pdays, previous, poutcome, y_text] = ...
 textread(filename, '%f %s %s %s %s %f %s %s %s %f %s %f %f %f %f %s %s', ...
 'delimiter', ';', 'headerlines', 1);

% Convert y to binary (1/2)
y = zeros(length(y_text), 1);
for i = 1:length(y_text)
    if strcmp(strtrim(y_text{i}), '"yes"') || strcmp(strtrim(y_text{i}), 'yes')
        y(i) = 2;  % positive class
    else
        y(i) = 1;  % negative class
    end
end

% Initialize feature matrices
X_numerical = zeros(length(y), 7);
X_encoded = [];
numerical_feature_names = {'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'};
encoded_feature_names = {};

% Fill numerical features
X_numerical(:, 1) = age;
X_numerical(:, 2) = balance;
X_numerical(:, 3) = day;
X_numerical(:, 4) = duration;
X_numerical(:, 5) = campaign;
X_numerical(:, 6) = pdays;
X_numerical(:, 7) = previous;

% Process categorical variables
% Function to one-hot encode a categorical variable
function [encoded, names] = oneHotEncode(variable, var_name)
    unique_values = unique(variable);
    encoded = zeros(length(variable), length(unique_values)-1);
    names = {};

    for j = 1:length(unique_values)-1
        value = unique_values{j};
        names{j} = [var_name, '_', strtrim(value)];
        for i = 1:length(variable)
            if strcmp(strtrim(variable{i}), value)
                encoded(i, j) = 1;
            end
        end
    end
end

% Process job
[job_encoded, job_names] = oneHotEncode(job, 'job');
X_encoded = [X_encoded, job_encoded];
encoded_feature_names = [encoded_feature_names, job_names];

% Process marital
[marital_encoded, marital_names] = oneHotEncode(marital, 'marital');
X_encoded = [X_encoded, marital_encoded];
encoded_feature_names = [encoded_feature_names, marital_names];

% Process education
[education_encoded, education_names] = oneHotEncode(education, 'education');
X_encoded = [X_encoded, education_encoded];
encoded_feature_names = [encoded_feature_names, education_names];

% Process default
[default_encoded, default_names] = oneHotEncode(default, 'default');
X_encoded = [X_encoded, default_encoded];
encoded_feature_names = [encoded_feature_names, default_names];

% Process housing
[housing_encoded, housing_names] = oneHotEncode(housing, 'housing');
X_encoded = [X_encoded, housing_encoded];
encoded_feature_names = [encoded_feature_names, housing_names];

% Process loan
[loan_encoded, loan_names] = oneHotEncode(loan, 'loan');
X_encoded = [X_encoded, loan_encoded];
encoded_feature_names = [encoded_feature_names, loan_names];

% Process contact
[contact_encoded, contact_names] = oneHotEncode(contact, 'contact');
X_encoded = [X_encoded, contact_encoded];
encoded_feature_names = [encoded_feature_names, contact_names];

% Process month
[month_encoded, month_names] = oneHotEncode(month, 'month');
X_encoded = [X_encoded, month_encoded];
encoded_feature_names = [encoded_feature_names, month_names];

% Process poutcome
[poutcome_encoded, poutcome_names] = oneHotEncode(poutcome, 'poutcome');
X_encoded = [X_encoded, poutcome_encoded];
encoded_feature_names = [encoded_feature_names, poutcome_names];

% Combine numerical and encoded features
X = [X_numerical, X_encoded];
feature_names = [numerical_feature_names, encoded_feature_names];

% Normalize numerical features
X_norm = X;
for i = 1:length(numerical_feature_names)
    col_idx = i;
    X_norm(:, col_idx) = (X(:, col_idx) - mean(X(:, col_idx))) ./ std(X(:, col_idx));
end

% Handle NaN values (replace with 0)
X_norm(isnan(X_norm)) = 0;

disp(['Preprocessed dataset: ', num2str(size(X_norm, 1)), ' samples, ', ...
      num2str(size(X_norm, 2)), ' features']);
disp(['Target distribution: ', num2str(sum(y == 2)), ' positive, ', ...
      num2str(sum(y == 1)), ' negative']);

end
