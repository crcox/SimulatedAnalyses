%% Iterative Lasso on small data set
clear;clc

%% Simulate a data set
ntrials = 200;
nvoxels = 200;
X = randn(ntrials,nvoxels);   % noise
size(X);

% Add some signals
X(1:ntrials/2,1) = X(1:ntrials/2,1) + 1;
X(ntrials/2 + 1:end,2) = X(ntrials/2 + 1 : end ,2) + 1;
% Add row label
rowLabels = zeros(ntrials,1);
rowLabels(1:ntrials /2 ,1) = 1;
numRL = 2; % number of labels

%% Cross validation
% Let's do 5 folds CV. In the 1st CV, I will take out trials 1:10 and
% trials 51:60. Train the model on the rest of the voxels. And test the
% model on the holdout set. 

%% Subset: 1st CV block
% K-folds
k = 5;
block = ntrials /k /numRL;

% test set
Xtest1 = X(1:block,:);
Xtest2 = X((ntrials / numRL) +1 : ntrials / numRL + block,:);
Xtest = vertcat(Xtest1, Xtest2); 
size_test = size(Xtest);

% training set
Xtrain1 = X(block + 1: ntrials / numRL, :);
Xtrain2 = X(ntrials / numRL + block + 1: end, :);
Xtrain = vertcat(Xtrain1,Xtrain2); 

% new labels (2 labels condition)
rowLabelsTrain = zeros(ntrials - size_test(1),1); 
rowLabelsTrain(1: (ntrials - size_test(1))/2 ,1) = 1; 
rowLabelsTest = zeros(size_test(1),1); 
rowLabelsTest(1:size_test(1) / 2 ,1) = 1; 


%% Analysis
% Fit LASSO
fit = glmnet(Xtrain, rowLabelsTrain, 'binomial')  

% Results on training set, which can be ignored
(Xtrain * fit.beta) + repmat(fit.a0, ntrials - size_test(1),1);     
% how well does the prediction fits the truth
((Xtrain * fit.beta) + repmat(fit.a0, ntrials - size_test(1),1)) > 0 

fit.df'                                  % How many voxels were selected for each lambda
fit.lambda                              % lambda values

predic_train = (Xtrain * fit.beta) + repmat(fit.a0, ntrials - size_test(1),1) > 0   % prediction
repmat(rowLabelsTrain,1,100) == predic_train          % compare prediction with truth
acc_train = mean(repmat(rowLabelsTrain,1,100) == predic_train)'   % accuracy

% glmnetPlot(fit)
% imagesc(fit.beta)       % two ways of visualize voxel selection process

% Results on testing set
Xtest*fit.beta + repmat(fit.a0, size_test(1), 1)          % prediction
predic_test = (Xtest*fit.beta + repmat(fit.a0, size_test(1), 1))> 0     % How well does it fits the truth
predic_test == repmat(rowLabelsTest,1,100)
acc_test = mean(repmat(rowLabelsTest,1,100) == predic_test)' 

max_acc = max(acc_test)
max_ind = find(acc_test == max(acc_test)); % store the indices for the best lambdas


%% Visualizing the results
plot(acc_train, 'Linewidth', 2)
hold on 
plot(acc_test, 'r', 'Linewidth', 2) 
max_xrange = 0: 0.5: 100;
plot(max_xrange,max_acc, 'g', 'Linewidth', 2)
chance_range = 0: 0.5 : 100;
plot(chance_range, 0.5,'k', 'linewidth', 2)
hold off

axis([1 100 0 1])
legend('accuary train', 'accuracy test','max for accuracy test',...
    'Location','SouthEast')
xlabel('Trails')
ylabel('Accuracy (chance = 0.5)')