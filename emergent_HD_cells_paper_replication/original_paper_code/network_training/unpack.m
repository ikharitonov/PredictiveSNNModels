function [Whx, Whh, Wyh, bah, bay, Tau] = unpack(parameters,dimIN,numh,dimOUT,Whx,Whh,Wyh,bah,bay,Tau,LEARNPARAMETERS_Whx,LEARNPARAMETERS_Whh,LEARNPARAMETERS_Wyh,LEARNPARAMETERS_bah,LEARNPARAMETERS_bay,LEARNPARAMETERS_Tau)
% parameters - parameters to be learned, some or all of Whx, Whh, Wyh, bah, bay, Tau depending on LEARNPARAMETERS
% Whx - numh x dimIN matrix
% Whh - numh x numh matrix
% Wyh - dimOUT x numh matrix
% bah  - numh x 1 matrix
% bay  - dimOUT x 1 matrix
% Tau  - numh x 1 matrix

% parameters that can be learned depending on LEARNPARAMETERS - Whx, Whh, Wyh, bah, bay, Tau
% For example, if LEARNPARAMETERS_Whx==1 overwrite Whx passed in explicitly to function with the values in the vector parameters 
i = 1;
if LEARNPARAMETERS_Whx==1; Whx = reshape(parameters(i:i+numh*dimIN-1),numh,dimIN); i = 1 + numh*dimIN; end
if LEARNPARAMETERS_Whh==1; Whh = reshape(parameters(i:i+numh*numh-1),numh,numh); i = i + numh*numh; end
if LEARNPARAMETERS_Wyh==1; Wyh = reshape(parameters(i:i+dimOUT*numh-1),dimOUT,numh); i = i + dimOUT*numh; end
if LEARNPARAMETERS_bah==1; bah = reshape(parameters(i:i+numh-1),numh,1); i = i + numh; end
if LEARNPARAMETERS_bay==1; bay = reshape(parameters(i:i+dimOUT-1),dimOUT,1); i = i + dimOUT; end
if LEARNPARAMETERS_Tau==1; Tau = reshape(parameters(i:i+numh-1),numh,1); i = i + numh; end