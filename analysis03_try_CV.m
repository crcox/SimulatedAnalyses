%% Iterative Lasso on small data set
clear;clc

%% Simulate a data set
ntrials = 100;
nvoxels = 200;
X = randn(ntrials,nvoxels);   % noise
size(X);

% Add some signals
X(1:50,1) = X(1:50,1) + 1;
X(51:100,2) = X(51:100,2) + 1;
% Add row label
rowLabels = zeros(100,1);
rowLabels(1:50,1) = 1;

%% Cross validation
% Let's do 5 folds CV. In the 1st CV, I will take out trials 1:10 and
% trials 51:60. Train the model on the rest of the voxels. And test the
% model on the holdout set. 

%% Subset: 1st CV block
% training set
Xtrain1 = X(11:50, :);
Xtrain2 = X(61:100, :);
Xtrain = vertcat(Xtrain1,Xtrain2);
% test set
Xtest1 = X(1:10,:);
Xtest2 = X(51:60,:);
Xtest = vertcat(Xtest1, Xtest2);
% new labels
rowLabelsTrain = zeros(80,1);
rowLabelsTrain(1:40,1) = 1;
rowLabelsTest = zeros(20,1);
rowLabelsTrain(1:10,1) = 1;


%% Analysis

% Fit LASSO
fit = glmnet(Xtrain, rowLabelsTrain, 'binomial')  

% Results on training set, which can be ignored
(Xtrain * fit.beta) + repmat(fit.a0, 80,1)       % prediction
((Xtrain * fit.beta) + repmat(fit.a0, 80,1)) > 0 % how well does the prediction fits the truth

fit.df'                                  % How many voxels were selected for each lambda
fit.lambda                              % lambda values

predic_train = (Xtrain * fit.beta) + repmat(fit.a0, 80,1) > 0   % prediction
repmat(rowLabelsTrain,1,100) == predic_train          % compare prediction with truth
acc_train = mean(repmat(rowLabelsTrain,1,100) == predic_train)'   % accuracy

% glmnetPlot(fit)
% imagesc(fit.beta)       % two ways of visualize voxel selection process

% Results on testing set
Xtest*fit.beta + repmat(fit.a0, 20, 1)          % prediction
predic_test = (Xtest*fit.beta + repmat(fit.a0, 20, 1))> 0     % How well does it fits the truth
predic_test == repmat(rowLabelsTest,1,100)
acc_test = mean(repmat(rowLabelsTest,1,100) == predic_test)' 

max_acc = max(acc_test)

%% Visualizing result
plot(acc_train, 'Linewidth', 2)
hold on 
plot(acc_test, 'r', 'Linewidth', 2) 
max_xrange = 0: 0.5: 100;
plot(max_xrange,max_acc, 'g', 'Linewidth', 2)
hold off


legend('accuary train', 'accuracy test','max for accuracy test',...
    'Location','SouthEast')
xlabel('Trails')
ylabel('Accuracy')