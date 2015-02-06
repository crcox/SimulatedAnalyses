%% Simulate 3D data
clear; clc; close all;



%% Create 3d coordinates 

% Set some parameters
axis.x = 10;
axis.y = 10;
axis.z = 10;
numTrials = 300;
numVoxels = axis.x * axis.y * axis.z;

% Assign length for x y z 
XYZ = expand_grid(1:axis.x,1:axis.y,1:axis.z);

% Create the data set 
X = zeros(numVoxels, numTrials);



%% Adding signal to a cluster
% Set the range of the cluster
signal.XYZ = expand_grid(2:4,2:4,2:4);
% Get the indices for the signals
signal.Ind = collapse_grid([axis.x,axis.y,axis.z], signal.XYZ);


% Visualize it, make sure they are doing the right thing
figure(2)
hold on 
scatter3(XYZ(:,1), XYZ(:,2), XYZ(:,3), 'b');
scatter3(signal.XYZ(:,1),signal.XYZ(:,2),signal.XYZ(:,3), 'r');
hold off



%% Adding noise 

%% Analysis




