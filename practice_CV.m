%% Practice generating CV indices
clear;clc;

%% Set the parameters for CV
% Dimension of the data sets
ntrials = 200;
nvoxels = 100;
% Number of folds
k = 5;


%% Create the indices matrix
A = 1:ntrials;
X = repmat(A', 1, nvoxels);

indices = crossvalind('Kfold', ntrials, k);

for i = 1:k

    
test = indices == i;
train = ~ test;

test = find(test);
train = find(train);

disp(['This is the ' num2str(i) 'th CV, the test set indices are: '])
disp(test')

Xtest = X(test, :);
Xtrain = X(train, :);

end