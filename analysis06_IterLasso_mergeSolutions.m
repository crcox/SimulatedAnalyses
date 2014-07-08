%% Iterative Lasso 
% This goal for this version is to complete iterative Lasso
% So it has to be able to merge all solutions 


%% WARNING: This will clear the work space & variables.
clear;clc;
w = warning ('off','all'); % somehow it returns a lot of warning

%% You can set the parameters for CV here
% Please the dimension of the data sets 
ntrials = 150; % it has to be divisible by K
nvoxels = 150;
% Please set the number of folds 
k = 5;
% It needs to know the number of row labels
% ps:Currently, this program can just run normally when there are 2 rowlabels
rowLabels.num = 2;

% Set the strength of the signal 
signal = 1;
numsignal = 10;
% Set the strength of the noise
rng(1) 
noise = 2;

% It is useful to know the size for the testing set
test.size = ntrials / k ;

% Display the parameters
disp ('Parameters: ')
disp(['number of Voxels= '  num2str(nvoxels)])
disp(['number of Trials= '  num2str(ntrials)])
disp(['K = '  num2str(k) ' (for K-folds CV)' ])
disp(['Number of row labels= ' num2str(rowLabels.num)])
disp(['Signal intensity = ' num2str(signal)])
disp(['Noise intensity = ' num2str(noise)])
disp(['Number of signals = ' num2str(numsignal)])

%% Simulate the data
% Creating the background 
X.raw = zeros(ntrials, nvoxels);

% % Add some signals 
X.raw(1:ntrials/rowLabels.num,1:numsignal / 2) = X.raw(1:ntrials/rowLabels.num,1:numsignal/2) + signal;
X.raw(ntrials/rowLabels.num + 1:end, numsignal/2 + 1 : numsignal) ...
    = X.raw(ntrials/rowLabels.num + 1 : end ,numsignal/2 + 1 : numsignal) + signal;
% plot the signal
figure(1)
imagesc(X.raw)
xlabel('Voxels');ylabel('Trials');title('Signal');

% Adding noise 
X.raw = X.raw + noise * randn(ntrials,nvoxels);   
% plot the noise + signal
figure(2)
imagesc(X.raw)
xlabel('Voxels');ylabel('Trials');title('Signal & Noise')

% Create row labels
rowLabels.whole = zeros(ntrials,1);
rowLabels.whole(1:ntrials / rowLabels.num ,1) = 1;
% imagesc(rowLabels.whole);

% Create new row labels for CV blocks
rowLabels.train = zeros(ntrials - test.size,1); 
rowLabels.train(1: (ntrials - test.size)/rowLabels.num ,1) = 1; 
rowLabels.test = zeros(test.size,1); 
rowLabels.test(1:test.size / rowLabels.num ,1) = 1; 



%% Generating indices for Outer CV
% generate indices for CV
CV.blocks = 1:k;
CV.blocks = CV.blocks';
CV.indices = repmat(CV.blocks,[ntrials / k,1]);

% Generate indices for CVglmnet
CV2.blocks = 1:k-1;
CV2.blocks = CV2.blocks';
CV2.indices = repmat(CV2.blocks,[ntrials / k,1]);

% Generate indices for CV (after the test set was taken out)
CV2.blocks = 1 : k-1;
CV2.blocks = CV2.blocks';
CV2.indices = repmat(CV2.blocks,[ntrials / k,1]);


% Creating indices for test set and training set
for i = 1: k
    test.indices(:,i) = CV.indices == i;
    train.indices(:,i) = CV.indices ~= i;
end



%% Prepare for Iterative Lasso

% Create a copy for the raw data (X.raw)
% btw, I am not sure I need it...
X.copy = X.raw;
% Initialize the remaining voxels. (At 1st time, all voxels are remain)
for i = 1:k
    voxel(i).remain = (1:nvoxels)';
end
% Keeping track of the number of iteration
numIter = 1;


%% Interative Lasso

% Create a matrix to index voxels that have been used (Chris' method)
used = false(k,nvoxels); 

% Counter for Stopping criterion 
% ps: the loop stops when t-test insig. twice
counter = 0;

% stopping 
while true
    
% Start 5 folds cross validation
    for i = 1:k

        % Re-subset data every time
        X.iter = X.raw(:,voxel(i).remain);

        % Split the data into training & testing set 
        X.test = X.iter(test.indices(:,i) ,:);
        X.train = X.iter(train.indices(:,i) ,:);

        % Fit cvglmnet
        cvfit(i) = cvglmnet (X.train, rowLabels.train, 'binomial', 'class', CV2.indices', 4);

        % Get the indice for the lambda with the best accuracy 
        lambda.best(i) = find(cvfit(i).lambda == cvfit(i).lambda_min);
        % Plot the cross-validation curve
%         cvglmnetPlot(cvfit(i));

        % Set the lambda value, using the numerical best
        opts(i) = glmnetSet();
        opts(i).lambda = cvfit(i).lambda_min;

        % Fit glmnet
        fit(i) = glmnet(X.train, rowLabels.train, 'binomial', opts);

        % Evaluate the prediction 
        %   test.prediction = X.test * ß (weight) + a (intercept)
        test.prediction(:,i) = (X.test * fit(i).beta + repmat(fit(i).a0, [test.size, 1])) > 0 ;  
        test.accuracy(:,i) = mean(rowLabels.test == test.prediction(:,i))';

        % Find indices for the voxels that have been used
        voxel(i).used = find (fit(i).beta ~= 0);
        % Find indices for the voxels that have not been used
        voxel(i).remain = find (fit(i).beta == 0);   
        
        % Recording voxels that have been used (chris' method)
        used( i, ~used(i,:) ) = fit(i).beta ~= 0;
        

        
    end
    
    
    for i = 1:5
    % Record the results, including 
    % 1) hit.num: how many voxels have been selected
    % 2) hit.rate: the porprotion of voxels have been selected 
    % 3) hit.accuracy: the accuracy for the correspoinding cv block
        hit.num(numIter,i) = sum(used(i,1:numsignal)); 
        hit.rate(numIter,i) = sum(used(i,1:numsignal)) / numsignal; 
        hit.accuracy(numIter, i) = test.accuracy(i);
    end

        
    %% Printing some results
    disp('==============================')
    
    % Keep track of the number of iteration.
    disp(['Iteration number: ' num2str(numIter)]);
    numIter = numIter + 1;
    
    % Display the average accuracy for this procedure 
    disp(['The accuracy for each CV: ' num2str(test.accuracy) ] );
    disp(['The mean accuracy: ' num2str(mean(test.accuracy))]);
    % Test classification accuracy aganist chance 
    [t,p] = ttest(test.accuracy, 0.5);
    disp(['Result for the t-test: ' num2str(t) ',  P value: ' num2str(p)]) 

    % Stop iteration, when the decoding accuracy is not better than chance
    if ttest(test.accuracy, 0.5) == 0
        counter = counter + 1;
 
        if counter == 2
        % stop, if t-test = 0 twice
            disp('==============================')
            disp('Iterative Lasso was terminated, as the classification accuracy is at chance level.')
            numIter = numIter - 1;   % adjust iteration number        
            break
        else 
            counter = 0;
            
        end

    end 
    
end

