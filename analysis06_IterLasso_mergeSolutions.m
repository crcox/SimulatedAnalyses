%% Iterative Lasso 
% This goal for this version is to complete iterative Lasso
% So it has to be able to merge all solutions 


%% WARNING: This will clear the work space & variables.
clear;clc;
w = warning ('off','all'); % somehow it returns a lot of warning
rng(1) % Set the seed (for reproducibility and debugging)

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
numsignal = 40;
% Set the strength of the noise

noise = 1.5;

% It is useful to know the size for the testing set
test.size = ntrials / k ;

% Display the parameters
disp ('Parameters: ')
disp(['number of Voxels = '  num2str(nvoxels)])
disp(['number of Trials = '  num2str(ntrials)])
disp(['K = '  num2str(k) '   (for K-folds CV)' ])
disp(['Number of row labels = ' num2str(rowLabels.num)])
disp(['Signal intensity = ' num2str(signal)])
disp(['Noise intensity = ' num2str(noise)])
disp(['Number of signal carrying voxels = ' num2str(numsignal)])
disp(' ')

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
numIter = 0;
% Counter for Stopping criterion 
% ps: the loop stops when t-test insig. twice
counter = 0;
% Create a matrix to index voxels that have been used (Chris' method)
used = false(k,nvoxels); 



%% Interative Lasso
 
% Infinite loop, until reach stopping crterion
while true
    
    numIter = numIter + 1;
    
% Start 5 folds cross validation
    for i = 1:k

        % Re-subset data every time
        X.iter = X.raw(:,~used(i,:));

        % Split the data into training & testing set 
        X.test = X.iter(test.indices(:,i) ,:);
        X.train = X.iter(train.indices(:,i) ,:);

        % Fit cvglmnet
        cvfit(i) = cvglmnet (X.train, rowLabels.train, 'binomial', 'class', CV2.indices', 4);

        % Plot the cross-validation curve
%         cvglmnetPlot(cvfit(i));

        % Set the lambda value, using the numerical best
        opts(i) = glmnetSet();
        opts(i).lambda = cvfit(i).lambda_min;

        % Fit glmnet
        fit(i) = glmnet(X.train, rowLabels.train, 'binomial', opts);

        % Evaluate the prediction 
        %   test.prediction = X.test * ß (weight) + a (intercept)
%         bsxfun(@plus, X.test * fit(i).beta, fit(i).a0) > 0 ;
        test.prediction(:,i) = (X.test * fit(i).beta + repmat(fit(i).a0, [test.size, 1])) > 0 ;  
        test.accuracy(:,i) = mean(rowLabels.test == test.prediction(:,i))';

        
        % Recording voxels that have been used (chris' method)
        used( i, ~used(i,:) ) = fit(i).beta ~= 0;
        
    end
    
    
    % Take a snapshot, find out which voxels were being used currently
    USED{numIter} = used;
    
    % Record the results, including 
    % 1) hit.num: how many voxels that carrying true signal have been selected
    % 2) hit.rate: the porprotion of voxels have been selected 
    % 3) hit.accuracy: the accuracy for the correspoinding cv block
    % 4) hit.all : how many voxel that have been selected
    hit.num(numIter,:) = sum(used(:,1:numsignal),2); 
    hit.rate(numIter,:) = sum(used(:,1:numsignal),2) / numsignal; 
    hit.accuracy(numIter, :) = test.accuracy;
    hit.all(numIter, :) = sum(used,2);
    
        
    %% Printing some results
    disp('==============================')
    
    % Keep track of the number of iteration.
    disp(['Iteration number: ' num2str(numIter)]);
    
    % Display the average accuracy for this procedure 
%     disp(['The accuracy for each CV: ' num2str(test.accuracy) ] );
    disp(['The mean accuracy: ' num2str(mean(test.accuracy))]);
    % Test classification accuracy aganist chance 
    [t,p] = ttest(test.accuracy, 0.5);
    disp(['Result for the t-test: ' num2str(t) ',  P value: ' num2str(p)]) 

    % Stop iteration, when the decoding accuracy is not better than chance
    if t ~= 1   %  ~t will rise a bug, when t is NaN
        counter = counter + 1;
 
        if counter == 2
        % stop, if t-test = 0 twice
            disp(' ')
            disp('* Iterative Lasso was terminated, as the classification accuracy is at chance level twice.')
            disp(' ')
            break
        end
        
    else
        counter = 0;
    end 

    
end

disp('Here are the accuracies for each iteration: ')
disp('(row: iteration, colum: CV)')
disp(hit.accuracy)
disp('Average:')
disp(mean(hit.accuracy,2)) 


% Plot the hit rate 
plot(hit.rate)
xlabel('Iterations');ylabel('Proportion');
title ('The Proportion of signal carrying voxels that were selected ');
axis([1 size(hit.rate(:,1),1) 0 1])
set(gca,'xtick',1:size(hit.rate(:,1),1))


%% Final step: pooling the solutions
disp('Pooling the solution, and fitting the final model...')
disp(' ')
for i = 1:k
    % Subset: find voxels that were selected 
    X.final = X.raw( :, USED{numIter - 2}(i,:) );

    % Split the final data set to testing set and training set 
    X.test = X.final(test.indices(:,i) ,:);
    X.train = X.final(train.indices(:,i) ,:);

    % Fit cvglmnet, in order to find the best lambda
    cvfitFinal = cvglmnet (X.train, rowLabels.train, 'binomial', 'class', CV2.indices', 4);

    % Set the lambda value, using the numerical best
    opts = glmnetSet();
    opts.lambda = cvfitFinal.lambda_min;
    opts.alpha = 0;

    % Fit glmnet 
    fitFinal = glmnet(X.train, rowLabels.train, 'binomial', opts);

    % Calculating accuracies
    final.prediction(:,i) = (X.test * fitFinal.beta + repmat(fitFinal.a0, [test.size, 1])) > 0 ;  
    final.accuracy(i) = mean(rowLabels.test == final.prediction(:,i))';


end

disp('Final accuracies: ')
disp('(row: CV that just performed, colum: CV block from the iterative Lasso)')
disp(final.accuracy)
disp('Average: ')
disp(mean(final.accuracy))
