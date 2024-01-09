function [ah0, h0, Whx, Whh, Wyh, bah, bay, Tau] = unpackall(parameters,dimIN,numh,dimOUT,numexamples)
% order of parameters
% ah0 - numh x numexamples matrix
% h0  - numh x numexamples matrix
% Whx - numh x dimIN matrix
% Whh - numh x numh matrix
% Wyh - dimOUT x numh matrix
% bah  - numh x 1 matrix
% bay  - dimOUT x 1 matrix
% Tau  - numh x 1 matrix

ah0 = -700*ones(numh,numexamples);
h0  = -700*ones(numh,numexamples);
Whx = -700*ones(numh,dimIN); 
Whh = -700*ones(numh,numh);
Wyh = -700*ones(dimOUT,numh);
bah  = -700*ones(numh,1);
bay  = -700*ones(dimOUT,1);
Tau  = -700*ones(numh,1);

ii = 1;
ah0(:) = parameters(ii:ii+numh*numexamples-1); ii = ii + numh*numexamples;
h0(:)  = parameters(ii:ii+numh*numexamples-1); ii = ii + numh*numexamples;
Whx(:) = parameters(ii:ii+numh*dimIN-1); ii = ii + numh*dimIN; 
Whh(:) = parameters(ii:ii+numh*numh-1); ii = ii + numh*numh; 
Wyh(:) = parameters(ii:ii+numh*dimOUT-1); ii = ii + dimOUT*numh; 
bah(:)  = parameters(ii:ii+numh-1); ii = ii + numh;
bay(:)  = parameters(ii:ii+dimOUT-1); ii = ii + dimOUT;
Tau(:)  = parameters(ii:ii+numh-1); ii = ii + numh;