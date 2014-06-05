%% Iterative Lasso on small data set

%% Simulate some data set
ntrials = 20;
nvoxels = 40;
X = randn(ntrials,nvoxels);
size(X);

% Add some signals
X(1:10,1) = X(1:10,1) + 1;
X(11:20,2) = X(11:20,2) + 1;

%% Analysis
% T-test, comparing 1st 10 trails with 2nd 10 traisl
h = ttest(X(1:10,:), X(11:20,:))

% Add row label
RowLabels = zeros(20,1);
RowLabels(1:10,1) = 1;

% Logistic Regression
[b,dev,stats] = glmfit(X, RowLabels, 'binomial', 'link', 'logit')

% Lasso
fit = glmnet(X, RowLabels, 'binomial')

(X * fit.beta) + repmat(fit.a0, 20,1)       % prediction
(X * fit.beta) + repmat(fit.a0, 20,1) > 0 % how well does the prediction fits the truth
(X * fit.beta(:,2)) + fit.a0(2)         % look at individual lamda
(X * fit.beta(:,2)) + fit.a0(2) > 0     % look at how this lamda fits the truth
fit.df'                                  % How many voxels were selected for each lambda
sum(abs(fit.beta)>0)                    % same as above
fit.lambda                              % lambda values

predictions = (X * fit.beta) + repmat(fit.a0, 20,1) > 0   % prediction
repmat(RowLabels,1,100) == predictions          % compare prediction with truth
mean(repmat(RowLabels,1,100) == predictions)'   % accuracy

imagesc(fit.beta)       % two ways of visualize voxel selection process
glmnetPlot(fit)