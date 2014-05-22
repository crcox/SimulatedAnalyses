%% Simulating fMRI data with some signal embedded
% The objective is to create a small dataset, that has some signal embedded
% within it, and then use a technique such as lasso to recover the signal
% you put there. For this, you are going to use a function for generating
% random data:
doc randn

ntrials = 20;
nvoxels = 40;
X = randn(ntrials,nvoxels);

size(X);

% This created a matrix filled with random numbers---but not just any
% random numbers. Random numbers samples from a normal distribution
% centered on zero with a standard deviation of one. This means that
% numbers closer to zero are more likely than numbers that are far from
% zero (either positive or negative.
q = -3:.01:3; % ``quantile''
m = 0;        % mean
s = 1;        % standard deviation
plot(q,pdf('norm',q,m,s))
% 1 SD from the mean
sd1 = pdf('norm',1,m,s);
line([1 1],[0,sd1],'Color',[1,0,0],'LineStyle','--');
line([-3 1],[sd1,sd1],'Color',[1,0,0],'LineStyle','--');

sd2 = pdf('norm',2,m,s);
line([2 2],[0,sd2],'Color',[0,1,0],'LineStyle','--');
line([-3 2],[sd2,sd2],'Color',[0,1,0],'LineStyle','--');

% So, the probability of a value 2 away from zero is about 0.05, and the
% probabilty of a number 1 away from zero is about .25 2 SD from the mean

% This is important to keep in mind, because understanding the distribution
% of the noise is key to having control over how strong the signal you
% create is.

% For a simple case, let's say the first voxel activates for the first 10
% items, and the second voxel activates for the seconds 10 items, and
% everything else is just left alone (i.e., activating randomly).

X(1:10,1) = X(1:10,1) + 1;
X(11:20,1) = X(11:20,1) + 1;

% Think for a moment---is this signal strong or weak? What is the
% probability of a value being 1 or greater, given the distribution of the
% noise? In the full dataset, what proportion of the voxels carry signal? 

% Think also---what do you think would be a good method for recovering this
% data? What if you just contrasted the activation of the first 10 rows
% with the activation of the second 10 rows for each column? 

% Before running any analyses, save your X matrix so that you can get back
% to it later. Read the manual if necessary.
doc save

% Now, for the sake of it, clear X from the workspace and load it back in.
doc clear
doc load

%  Now: run a t-test at each voxel. That is, for each voxel,
%  compare the activation for the first 10 rows to the second 10 rows.
%  Indicate any voxels that show a significant difference.
doc ttest
% or, if you understand the relationship between regression and t-tests
doc fitglm 

% This might be tricky to wrap your head around at first, but you can do a
% t-test with a regression model. The outcome variable is the activatin
% itself, and you are trying to predict that activation using the row
% labels (which you would need to generate).

%  Now try logistic regression. First---how will you need to set up this
%  problem? You will need an additional variable before you can do logistic
%  regression.
doc fitglm; 
% look under family for binomial. Logistic regression uses a binomial
% distribution.


%  As you might have expected, logistic regression doesn't work. Can you
%  explain why?

%  Now, give LASSO a shot. Use glmnet() to fit LASSO to your fake data.