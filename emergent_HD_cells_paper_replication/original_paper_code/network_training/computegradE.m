function [E, Emain, EL1, EL2, EhL1, EhL2, gradE] = computegradE(parameters,model)% CJ Cueva 5.12.2017, calling the function with no specified output will only return the first output
% compute the gradient of the error function
% no structural damping, assumes loss function matches output nonlinearity
%--------------------------------------------------------------------------
%                       recurrent neural network 
%      Tau*dah/dt = -ah + Whx*IN + Whh*h + bahneverlearn + bah
%--------------------------------------------------------------------------
% t=1 ah(:,1) = ah0       + (dt./Tau).*(-ah0       + Whx*IN(:,1) + Whh*h0       + bahneverlearn(:,t) + bah)       hidden activation
% t>1 ah(:,t) = ah(:,t-1) + (dt./Tau).*(-ah(:,t-1) + Whx*IN(:,t) + Whh*h(:,t-1) + bahneverlearn(:,t) + bah)       hidden activation
%      h(:,t) = f(ah(:,t)) + bhneverlearn(:,t)                             hidden units
%     ay(:,t) = Wyh*h(:,t) + bayneverlearn(:,t) + bay                      output activation
%      y(:,t) = g(ay(:,t))                                                 output units
%--------------------------------------------------------------------------
% INPUTS
% parameters        - parameters to be learned, some or all of Whx, Whh, Wyh, bah, bay, Tau depending on LEARNPARAMETERS
% LEARNPARAMETERS_W - if 1 learn W, if 0 treat as constant
% L1REGULARIZE_W    - if 1 regularize W with lambdaL1, larger lambdaL1 = more regularization = smaller parameters
% L2REGULARIZE_W    - if 1 regularize W with lambdaL2, larger lambdaL2 = more regularization = smaller parameters
% ah0 - numh x numexamples matrix, initial activation of hidden units
% h0  - numh x numexamples matrix, initial values of hidden units
% bahneverlearn - numh x numT x numexamples matrix, hidden activation bias
% bhneverlearn - numh x numT x numexamples matrix, hidden bias
% bayneverlearn - dimOUT x numT x numexamples matrix, output bias
% nonlinearity{1} - hidden unit nonlinearity, options: 'linear' 'logistic' 'tanh' 'retanh' 'ReLU' 
% nonlinearity{2} - output unit nonlinearity, options: 'linear' 'logistic' 'tanh' 'retanh' 'ReLU' 
% IN        - dimIN x numT x numexamples matrix, IN(:,i,j) is an input vector at time-step i for example j 
% TARGETOUT - dimOUT x numT x numexamples matrix, TARGETOUT(:,i,j) is the target output at time-step i for example j
% itimeRNN  - dimOUT x numT x numexamples matrix, elements 0(time-point does not contribute to first term in cost function),1(time-point contributes to first term in cost function)
% lambdaL1  - L1 regularization on parameters, larger lambdaL1 = more regularization = smaller parameters
% lambdaL2  - L2 regularization on parameters, larger lambdaL2 = more regularization = smaller parameters

% OUTPUTS
% E - 1 x 1 matrix, E = (1/sum(itimeRNN(:)==1)) * (y(:) - TARGETOUT(:))'*(y(:) - TARGETOUT(:))  +  L1 regularization on parameters  +  L2 regularization on parameters
% gradE - column vector that is the same size as parameters
%--------------------------------------------------------------------------
LEARNPARAMETERS_Whx = model.LEARNPARAMETERS_Whx;
LEARNPARAMETERS_Whh = model.LEARNPARAMETERS_Whh;
LEARNPARAMETERS_Wyh = model.LEARNPARAMETERS_Wyh;
LEARNPARAMETERS_bah = model.LEARNPARAMETERS_bah;
LEARNPARAMETERS_bay = model.LEARNPARAMETERS_bay;
LEARNPARAMETERS_Tau = model.LEARNPARAMETERS_Tau;
L1REGULARIZE_Whx = model.L1REGULARIZE_Whx;
L1REGULARIZE_Whh = model.L1REGULARIZE_Whh;
L1REGULARIZE_Wyh = model.L1REGULARIZE_Wyh;
L1REGULARIZE_bah = model.L1REGULARIZE_bah;
L1REGULARIZE_bay = model.L1REGULARIZE_bay;
L1REGULARIZE_Tau = model.L1REGULARIZE_Tau;
L2REGULARIZE_Whx = model.L2REGULARIZE_Whx;
L2REGULARIZE_Whh = model.L2REGULARIZE_Whh;
L2REGULARIZE_Wyh = model.L2REGULARIZE_Wyh;
L2REGULARIZE_bah = model.L2REGULARIZE_bah;
L2REGULARIZE_bay = model.L2REGULARIZE_bay;
L2REGULARIZE_Tau = model.L2REGULARIZE_Tau;
Whx = model.Whx;
Whh = model.Whh;
Wyh = model.Wyh;
bah = model.bah;% numh x 1 matrix
bay = model.bay;% dimOUT x 1 matrix
Tau = model.Tau;% numh x 1 matrix
ah0 = model.ah0;
h0 = model.h0;
bahneverlearn = model.bahneverlearn;% numh x numT x numexamples matrix
bhneverlearn = model.bhneverlearn;% numh x numT x numexamples matrix
bayneverlearn = model.bayneverlearn;% dimOUT x numT x numexamples matrix
dt = model.dt;
IN = model.IN;
TARGETOUT = model.TARGETOUT;
nonlinearity = model.nonlinearity;
itimeRNN = model.itimeRNN;
lambdaL1 = model.lambdaL1;% L1 regularization on parameters 
lambdaL2 = model.lambdaL2;% L2 regularization on parameters
lambdahL1 = model.lambdahL1;% L1 regularization on h
lambdahL2 = model.lambdahL2;% L2 regularization on h


[dimIN, numT, numexamples] = size(IN);
[dimOUT, numT, numexamples] = size(TARGETOUT);
numh = size(h0,1);
permutebahneverlearn = permute(bahneverlearn,[1 3 2]);% numh x numexamples x numT matrix, permute dimensions of array because squeeze(randn(1,5,1)) has dimensions 1 x 5 as opposed to squeeze(randn(1,1,5) which has dimensions 5 x 1
permutebhneverlearn = permute(bhneverlearn,[1 3 2]);% numh x numexamples x numT matrix, permute dimensions of array because squeeze(randn(1,5,1)) has dimensions 1 x 5 as opposed to squeeze(randn(1,1,5) which has dimensions 5 x 1
permutebayneverlearn = permute(bayneverlearn,[1 3 2]);% dimOUT x numexamples x numT matrix, permute dimensions of array because squeeze(randn(1,5,1)) has dimensions 1 x 5 as opposed to squeeze(randn(1,1,5) which has dimensions 5 x 1
permuteIN = permute(IN,[1 3 2]);% dimIN x numexamples x numT matrix, permute dimensions of array because squeeze(randn(1,5,1)) has dimensions 1 x 5 as opposed to squeeze(randn(1,1,5) which has dimensions 5 x 1
permuteTARGETOUT = permute(TARGETOUT,[1 3 2]);% dimOUT x numexamples x numT matrix, permute dimensions of array because squeeze(randn(1,5,1)) has dimensions 1 x 5 as opposed to squeeze(randn(1,1,5) which has dimensions 5 x 1
permuteitimeRNN = permute(itimeRNN,[1 3 2]);% dimOUT x numexamples x numT matrix
sumitimeRNN = sum(itimeRNN(:)==1);
if sumitimeRNN==0; sumitimeRNN = 1; end% dividing by 0 creates NaN so divide by 1 instead, this line would not be a good idea if sumitimeRNN was used to multiply other numbers because multiplying by 0 and 1 are very different!

% parameters that can be learned depending on LEARNPARAMETERS - Whx, Whh, Wyh, bah, bay, Tau
% For example, if LEARNPARAMETERS_Whx==1 overwrite Whx passed in explicitly to function with the values in the vector parameters 
i = 1;
if LEARNPARAMETERS_Whx==1; Whx = reshape(parameters(i:i+numh*dimIN-1),numh,dimIN); i = 1 + numh*dimIN; end
if LEARNPARAMETERS_Whh==1; Whh = reshape(parameters(i:i+numh*numh-1),numh,numh); i = i + numh*numh; end
if LEARNPARAMETERS_Wyh==1; Wyh = reshape(parameters(i:i+dimOUT*numh-1),dimOUT,numh); i = i + dimOUT*numh; end
if LEARNPARAMETERS_bah==1; bah = reshape(parameters(i:i+numh-1),numh,1); i = i + numh; end
if LEARNPARAMETERS_bay==1; bay = reshape(parameters(i:i+dimOUT-1),dimOUT,1); i = i + dimOUT; end
if LEARNPARAMETERS_Tau==1; Tau = reshape(parameters(i:i+numh-1),numh,1); i = i + numh; end

% forward pass
if isfield(model,'singleprecision')
    ah_store = zeros(numh,numexamples,numT,'single');% numh x numexamples x numT matrix
    h = zeros(numh,numexamples,numT,'single');% numh x numexamples x numT matrix
    %y = zeros(dimOUT,numexamples,numT,'single');% dimOUT x numexamples x numT matrix, outputs, each column of y is a desired output vector corresponding to the input vector given by the same column of x
    bahrepmat = bah*ones(1,numexamples,'single');% numh x numexamples matrix
    bayrepmat = bay*ones(1,numexamples,'single');% dimOUT x numexamples matrix
else
    ah_store = zeros(numh,numexamples,numT);% numh x numexamples x numT matrix
    h = zeros(numh,numexamples,numT);% numh x numexamples x numT matrix
    %y = zeros(dimOUT,numexamples,numT);% dimOUT x numexamples x numT matrix, outputs, each column of y is a desired output vector corresponding to the input vector given by the same column of x
    bahrepmat = bah*ones(1,numexamples);% numh x numexamples matrix
    bayrepmat = bay*ones(1,numexamples);% dimOUT x numexamples matrix
end
ahtminus = ah0;% numh x numexamples
htminus = h0;% numh x numexamples
dtoverTaurepmat = (dt./Tau)*ones(1,numexamples);% numh x numexamples matrix
if LEARNPARAMETERS_Tau==1; stuffforTau = zeros(numh,numexamples,numT); end
if isequal(bahneverlearn,0) && isequal(bhneverlearn,0) && isequal(bayneverlearn,0)
    for t=1:numT
        A = -ahtminus + Whx*permuteIN(:,:,t) + Whh*htminus + bahrepmat;% numh x numexamples
        ah = ahtminus + dtoverTaurepmat.*A;% numh x numexamples
        
        ah_store(:,:,t) = ah;
        if LEARNPARAMETERS_Tau; stuffforTau(:,:,t) = A; end
        h(:,:,t) = f(ah ,nonlinearity{1});% numh x numexamples
        %y(:,:,t) = f(Wyh*h(:,:,t) + bay ,nonlinearity{2});% dimOUT x numexamples, bay is broadcast
        ahtminus = ah;% numh x numexamples
        htminus = h(:,:,t);% numh x numexamples
    end% for t=1:numT
    y = f(pagemtimes(Wyh,h) + bay ,nonlinearity{2});% dimOUT x numexamples x numT matrix, pagemtimes computes Wyh*h(:,:,i) for each i, bay is broadcast
else
    for t=1:numT
        A = -ahtminus + Whx*permuteIN(:,:,t) + Whh*htminus + bahrepmat + permutebahneverlearn(:,:,t);% numh x numexamples
        ah = ahtminus + dtoverTaurepmat.*A;% numh x numexamples
        
        ah_store(:,:,t) = ah;
        if LEARNPARAMETERS_Tau; stuffforTau(:,:,t) = A; end
        h(:,:,t) = f(ah ,nonlinearity{1}) + permutebhneverlearn(:,:,t);% numh x numexamples
        %y(:,:,t) = f(Wyh*h(:,:,t) + bayrepmat + permutebayneverlearn(:,:,t) ,nonlinearity{2});% dimOUT x numexamples
        ahtminus = ah;% numh x numexamples
        htminus = h(:,:,t);% numh x numexamples
    end% for t=1:numT
    y = f(pagemtimes(Wyh,h) + bay + permutebayneverlearn,nonlinearity{2});% dimOUT x numexamples x numT matrix, pagemtimes computes Wyh*h(:,:,i) for each i, bay is broadcast
end
%[dhda, d2hda2] = df(h,nonlinearity{1});% numh x numexamples x numT matrix
[h_withoutbiases, dhda] = f(ah_store,nonlinearity{1});% numh x numexamples x numT matrix

% cost function
%-----------------
A = L(y(permuteitimeRNN==1),permuteTARGETOUT(permuteitimeRNN==1) ,nonlinearity{2});% matching loss function for nonlinearity{2}
Emain = sum(A(:)) / sumitimeRNN;
%-----------------
EL1 = 0;
numelementstoregularize_L1 = 0;
A = [];
if L1REGULARIZE_Whx==1 && LEARNPARAMETERS_Whx==1; A = [A; Whx(:)]; numelementstoregularize_L1 = numelementstoregularize_L1 + numel(Whx); end 
if L1REGULARIZE_Whh==1 && LEARNPARAMETERS_Whh==1; A = [A; Whh(:)]; numelementstoregularize_L1 = numelementstoregularize_L1 + numel(Whh); end 
if L1REGULARIZE_Wyh==1 && LEARNPARAMETERS_Wyh==1; A = [A; Wyh(:)]; numelementstoregularize_L1 = numelementstoregularize_L1 + numel(Wyh); end 
if L1REGULARIZE_bah==1 && LEARNPARAMETERS_bah==1; A = [A; bah]; numelementstoregularize_L1 = numelementstoregularize_L1 + numel(bah); end 
if L1REGULARIZE_bay==1 && LEARNPARAMETERS_bay==1; A = [A; bay]; numelementstoregularize_L1 = numelementstoregularize_L1 + numel(bay); end 
if L1REGULARIZE_Tau==1 && LEARNPARAMETERS_Tau==1; A = [A; Tau]; numelementstoregularize_L1 = numelementstoregularize_L1 + numel(Tau); end 
if numelementstoregularize_L1~=0; EL1 = (lambdaL1/numelementstoregularize_L1) * sum(abs(A)); end% if numelementstoregularize is 0 then computing this line makes EL1 = [] and Emain = []
%-----------------
EL2 = 0;
numelementstoregularize_L2 = 0;
A = [];
if L2REGULARIZE_Whx==1 && LEARNPARAMETERS_Whx==1; A = [A; Whx(:)]; numelementstoregularize_L2 = numelementstoregularize_L2 + numel(Whx); end 
if L2REGULARIZE_Whh==1 && LEARNPARAMETERS_Whh==1; A = [A; Whh(:)]; numelementstoregularize_L2 = numelementstoregularize_L2 + numel(Whh); end 
if L2REGULARIZE_Wyh==1 && LEARNPARAMETERS_Wyh==1; A = [A; Wyh(:)]; numelementstoregularize_L2 = numelementstoregularize_L2 + numel(Wyh); end 
if L2REGULARIZE_bah==1 && LEARNPARAMETERS_bah==1; A = [A; bah]; numelementstoregularize_L2 = numelementstoregularize_L2 + numel(bah); end 
if L2REGULARIZE_bay==1 && LEARNPARAMETERS_bay==1; A = [A; bay]; numelementstoregularize_L2 = numelementstoregularize_L2 + numel(bay); end 
if L2REGULARIZE_Tau==1 && LEARNPARAMETERS_Tau==1; A = [A; Tau]; numelementstoregularize_L2 = numelementstoregularize_L2 + numel(Tau); end 
if numelementstoregularize_L2~=0; EL2 = (lambdaL2/numelementstoregularize_L2) * (A' * A); end% if numelementstoregularize is 0 then computing this line makes EL2 = [] and Emain = []
%-----------------
EhL1 = (lambdahL1/(numexamples*numT*numh))*sum(abs(h(:)));
%-----------------
EhL2 = (lambdahL2/(numexamples*numT*numh))*(h(:)'*h(:));
%-----------------
E = Emain + EL1 + EL2 + EhL1 + EhL2; if nargout<=6; return; end% return E, Emain, EL1, EL2, EhL1, EhL2


% backward pass
deltay = y - permuteTARGETOUT;% dimOUT x numexamples x numT matrix, assumes loss function matches output nonlinearity
deltay(permuteitimeRNN==0) = 0;
delta = dhda .* reshape(Wyh'*deltay(:,:),numh,numexamples,numT)/sumitimeRNN  +...
        (lambdahL1/(numexamples*numT*numh)) * (dhda .* ((h > 0) - (h < 0))) +...
        (lambdahL2/(numexamples*numT*numh)) * 2*h .* dhda;% numh x numexamples x numT matrix, deltah + deltahL1 + deltahL2 
for t=numT-1:-1:1
    delta(:,:,t)    = delta(:,:,t)    + (1-dtoverTaurepmat).*delta(:,:,t+1)    + dhda(:,:,t) .* ( (((dt./Tau)*ones(1,numh)) .* Whh)'*delta(:,:,t+1));% numh x numexamples matrix
end


% compute gradients
dEdWhx = []; dEdWhh = []; dEdWyh = []; dEdbah = []; dEdbay = []; dEdTau = [];
A = delta(:,:);% numh x numexamples*numT matrix
if LEARNPARAMETERS_Whx==1; dEdWhx = ((dt./Tau)*ones(1,dimIN)) .* (A * permuteIN(:,:)'); end% numh x dimIN matrix
if LEARNPARAMETERS_Whh==1; dEdWhh = ((dt./Tau)*ones(1,numh))  .* (A * [h0 reshape(h(:,:,1:numT-1),numh,numexamples*(numT-1))]'); end% numh x numh matrix
if LEARNPARAMETERS_Wyh==1; dEdWyh = deltay(:,:) * h(:,:)' / sumitimeRNN; end% dimOUT x numh matrix
if LEARNPARAMETERS_bah==1; dEdbah = (dt./Tau) .* sum(A,2); end% numh x 1 matrix
if LEARNPARAMETERS_bay==1; dEdbay = sum(deltay(:,:),2) / sumitimeRNN; end% dimOUT x 1 matrix
if LEARNPARAMETERS_Tau==1; dEdTau = (-dt./(Tau.^2)) .* sum(A.*stuffforTau(:,:),2); end% numh x 1 matrix

if L1REGULARIZE_Whx==1 && LEARNPARAMETERS_Whx==1; dEdWhx = dEdWhx + (lambdaL1/numelementstoregularize_L1)*((Whx>0) - (Whx<0)); end 
if L1REGULARIZE_Whh==1 && LEARNPARAMETERS_Whh==1; dEdWhh = dEdWhh + (lambdaL1/numelementstoregularize_L1)*((Whh>0) - (Whh<0)); end 
if L1REGULARIZE_Wyh==1 && LEARNPARAMETERS_Wyh==1; dEdWyh = dEdWyh + (lambdaL1/numelementstoregularize_L1)*((Wyh>0) - (Wyh<0)); end 
if L1REGULARIZE_bah==1 && LEARNPARAMETERS_bah==1; dEdbah = dEdbah + (lambdaL1/numelementstoregularize_L1)*((bah>0) - (bah<0)); end 
if L1REGULARIZE_bay==1 && LEARNPARAMETERS_bay==1; dEdbay = dEdbay + (lambdaL1/numelementstoregularize_L1)*((bay>0) - (bay<0)); end 
if L1REGULARIZE_Tau==1 && LEARNPARAMETERS_Tau==1; dEdTau = dEdTau + (lambdaL1/numelementstoregularize_L1)*((Tau>0) - (Tau<0)); end 

if L2REGULARIZE_Whx==1 && LEARNPARAMETERS_Whx==1; dEdWhx = dEdWhx + (2*lambdaL2/numelementstoregularize_L2)*Whx; end 
if L2REGULARIZE_Whh==1 && LEARNPARAMETERS_Whh==1; dEdWhh = dEdWhh + (2*lambdaL2/numelementstoregularize_L2)*Whh; end 
if L2REGULARIZE_Wyh==1 && LEARNPARAMETERS_Wyh==1; dEdWyh = dEdWyh + (2*lambdaL2/numelementstoregularize_L2)*Wyh; end 
if L2REGULARIZE_bah==1 && LEARNPARAMETERS_bah==1; dEdbah = dEdbah + (2*lambdaL2/numelementstoregularize_L2)*bah; end 
if L2REGULARIZE_bay==1 && LEARNPARAMETERS_bay==1; dEdbay = dEdbay + (2*lambdaL2/numelementstoregularize_L2)*bay; end 
if L2REGULARIZE_Tau==1 && LEARNPARAMETERS_Tau==1; dEdTau = dEdTau + (2*lambdaL2/numelementstoregularize_L2)*Tau; end 
gradE = [dEdWhx(:); dEdWhh(:); dEdWyh(:); dEdbah; dEdbay; dEdTau];% numparameters x 1 matrix
end

%--------------------------------------------------------------------------
%          compute specified nonlinearity and its derivative
%--------------------------------------------------------------------------
function [F, dFdx] = f(x,type,varargin)% calling the function with no specified output will only return the first output (F)
switch type
    case 'linear'
        F = x;
        if nargout > 1; dFdx = ones(size(x)); end
    case 'logistic'
        F = 1 ./ (1 + exp(-x));
        if nargout > 1; dFdx = F - F.^2; end% dfdx = f(x)-f(x).^2 = F-F.^2
    case 'smoothReLU'% smoothReLU or softplus
        F = log(1 + exp(x));% always greater than zero
        if nargout > 1; dFdx = 1 ./ (1 + exp(-x)); end% dFdx = 1./(1 + exp(-x)) = 1 - exp(-F)    
    case 'ReLU'% rectified linear units
        F = max(x,0);
        if nargout > 1; dFdx = (x > 0); end
    case 'swish'% swish or SiLU (sigmoid linear unit)
        % Hendrycks and Gimpel 2016 "Gaussian Error Linear Units (GELUs)"
        % Elfwing et al. 2017 "Sigmoid-weighted linear units for neural network function approximation in reinforcement learning"
        % Ramachandran et al. 2017 "Searching for activation functions"
        sigmoid = 1./(1+exp(-x));
        F = x .* sigmoid;% x*sigmoid(x)
        if nargout > 1; dFdx = F + sigmoid .* (1 - F); end
    case 'mish'% Misra 2019 "Mish: A Self Regularized Non-Monotonic Neural Activation Function
        F = x .* tanh(log(1+exp(x)));
        if nargout > 1
            omega = 4*(x+1) + 4*exp(2*x) + exp(3*x) + exp(x).*(4*x+6);
            delta = 2*exp(x) + exp(2*x) + 2;
            dFdx = exp(x) .* omega ./ (delta.^2); 
        end
    case 'GELU'% Hendrycks and Gimpel 2016 "Gaussian Error Linear Units (GELUs)"
        F = 0.5 * x .* (1 + tanh(sqrt(2/pi)*(x + 0.044715*x.^3)));% fast approximating used in original paper
        %F = x.*normcdf(x,0,1);% x.*normcdf(x,0,1)  =  x*0.5.*(1 + erf(x/sqrt(2)))
        %figure; hold on; x = linspace(-5,5,100); plot(x,x.*normcdf(x,0,1),'k-'); plot(x,0.5*x.*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x.^3))),'r--')
        if nargout > 1
            A = tanh(sqrt(2/pi)*(x + 0.044715*x.^3));
            dFdx = 0.5*(1+A) + 0.5*x.*(1-A.^2).*sqrt(2/pi).*(1+0.044715*3*x.^2);
        end           
    case 'ELU'% exponential linear units, Clevert et al. 2015 "FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)"
        alpha = 1;
        inegativex = (x < 0);
        F = x; F(inegativex) = alpha * (exp(x(inegativex)) - 1);
        if nargout > 1; dFdx = ones(size(F)); dFdx(inegativex) = F(inegativex) + alpha; end 
    case 'tanh'
        F = tanh(x);
        if nargout > 1; dFdx = 1 - F.^2; end% dfdx = 1-f(x).^2 = 1-F.^2
    case 'tanhwithslope'
        a = varargin{1};
        F = tanh(a*x);
        if nargout > 1; dFdx = a - a*(F.^2); end% F(x)=tanh(a*x), dFdx=a-a*(tanh(a*x).^2), d2dFdx=-2*a^2*tanh(a*x)*(1-tanh(a*x).^2)     
    case 'tanhlecun'% LeCun 1998 "Efficient Backprop" 
        F = 1.7159*tanh(2/3*x);% F(x)=a*tanh(b*x), dFdx=a*b-a*b*(tanh(b*x).^2), d2dFdx=-2*a*b^2*tanh(b*x)*(1-tanh(b*x).^2) 
        if nargout > 1; dFdx = 1.7159*2/3 - 2/3*(F.^2)/1.7159; end
    case 'lineartanh'
        F = min(max(x,-1),1);% -1(x<-1), x(-1<=x<=1), 1(x>1)
        if nargout > 1; dFdx = ((x>-1) .* (x<1)) + 0.5*(x==-1) + 0.5*(x==1); end% 0(x<-1), 1(-1<=x<=1), 0(x>1)
    case 'retanh'
        F = max(tanh(x),0);
        if nargout > 1; dFdx = (1 - F.^2) .* (x > 0); end% dfdx = 1-f(x).^2 = 1-F.^2  
    case 'binarymeanzero'% binary units with output values -1 and +1
        F = (x>=0) - (x<0);
        if nargout > 1; dFdx = zeros(size(F)); end    
    otherwise
        error('Unknown transfer function type');
end
end

function [dFdx, d2Fdx2] = df(F,type,varargin)% input has already been passed through nonlinearity, F = f(x), calling the function with no specified output will only return the first output (dFdx)
switch type
    case 'linear'
        dFdx = ones(size(F)); 
        if nargout > 1; d2Fdx2 = zeros(size(F)); end
    case 'logistic'
        dFdx = F - F.^2;% dfdx = f(x)-f(x).^2 = F-F.^2
        if nargout > 1; d2Fdx2 = F.*(1-F).*(1-2*F); end
    case 'smoothReLU'% smoothReLU or softplus
        dFdx = 1 - exp(-F);% dFdx = 1./(1 + exp(-x)) = 1 - exp(-F)
        if nargout > 1; d2Fdx2 = dFdx - dFdx.^2; end
    case 'ReLU'% rectified linear units
        dFdx = (F > 0);% F > 0 is the same as x > 0 for ReLU nonlinearity  
        if nargout > 1; d2Fdx2 = zeros(size(F)); end
    case 'ELU'% exponential linear units, Clevert et al. 2015 "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)"
        alpha = 1;
        inegativex = (F < 0);% F < 0 is the same as x < 0 for ELU nonlinearity
        dFdx = ones(size(F)); dFdx(inegativex) = F(inegativex) + alpha;
        if nargout > 1; d2Fdx2 = zeros(size(F)); d2Fdx2(inegativex) = dFdx(inegativex); end    
    case 'tanh'
        dFdx = 1 - F.^2;% dfdx = 1-f(x).^2 = 1-F.^2
        if nargout > 1; d2Fdx2 = -2*F.*(1-F.^2); end
    case 'tanhwithslope'
        a = varargin{1};
        dFdx = a - a*(F.^2);% F(x)=tanh(a*x), dFdx=a-a*(tanh(a*x).^2), d2dFdx=-2*a^2*tanh(a*x)*(1-tanh(a*x).^2)    
        if nargout > 1; d2Fdx2 = -2*a^2*F.*(1-(F.^2)); end     
    case 'tanhlecun'% LeCun 1998 "Efficient Backprop" 
        dFdx = 1.7159*2/3 - 2/3*(F.^2)/1.7159;% F(x)=a*tanh(b*x), dFdx=a*b-a*b*(tanh(b*x).^2), d2dFdx=-2*a*b^2*tanh(b*x)*(1-tanh(b*x).^2)    
        if nargout > 1; d2Fdx2 = -2*(2/3)^2*F.*(1-(F.^2)/(1.7159^2)); end        
    case 'lineartanh'
        dFdx = ((F>-1) .* (F<1));% 0(F<=-1), 1(-1<F<1), 0(F>=1), not quite right at x=-1 and x=1
        if nargout > 1; d2Fdx2 = zeros(size(F)); end
    case 'retanh'
        dFdx = (1 - F.^2) .* (F > 0);% dfdx = 1-f(x).^2 = 1-F.^2
        if nargout > 1; d2Fdx2 = -2*F.*(1-F.^2) .* (F > 0); end% F > 0 is the same as x > 0 for tanh nonlinearity
    case 'binarymeanzero'% binary units with output values -1 and +1
        dFdx = zeros(size(F));
        if nargout > 1; d2Fdx2 = zeros(size(F)); end    
    otherwise
        error('Unknown transfer function type');
end
end

function [o] = L(y,ytarget,type)% L is the matching loss function for the specified nonlinearity
switch type
    case 'linear'
        o = 0.5*(y-ytarget).^2;
    case 'logistic'
        o = -ytarget.*log(y)-(1-ytarget).*log(1-y);% cross-entropy cost function   
    case 'tanh'
        o = -0.5*(1-ytarget).*log(1-y) -0.5*(1+ytarget).*log(1+y); 
    otherwise
        error('Unknown transfer function type');
end
end
