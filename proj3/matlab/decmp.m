clear all; close all; clc;
addpath('nmfv1_4');

disp('Reading data...');
R=dlmread('out.csv');
% R(R == 0) = nan;


% Weight
W=(R==0);

%W=isnan(R);
%R(W)=0;
W=~W;

disp('Doing decomposition...');
for k=[10,50,100]
    disp(['k=',num2str(k)])
%     [U,V,numIter,tElapsed,finalResidual]=nmfrule(R,k);
    [U,V,numIter,tElapsed,finalResidual]=wnmfrule(R,W,k);
%     D = abs(R-U*V).^2;
%     tse = sum(D(:));
%     disp(['total square error = ', num2str(tse)]);
end
