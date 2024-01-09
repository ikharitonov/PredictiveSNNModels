function [ah, h, ay, y, stuffforTau] = forwardpass(Whx,Whh,Wyh,bah,bay,Tau,ah0,h0,bahneverlearn,bhneverlearn,bayneverlearn,dt,IN,nonlinearity)% CJ Cueva 7.11.2017
%--------------------------------------------------------------------------
%                       recurrent neural network 
%--------------------------------------------------------------------------
% t=1 ah(:,1) = ah0       + (dt./Tau).*(-ah0       + Whx*IN(:,1) + Whh*h0       + bahneverlearn(:,t) + bah)       hidden activation
% t>1 ah(:,t) = ah(:,t-1) + (dt./Tau).*(-ah(:,t-1) + Whx*IN(:,t) + Whh*h(:,t-1) + bahneverlearn(:,t) + bah)       hidden activation
%      h(:,t) = f(ah(:,t)) + bhneverlearn(:,t)                             hidden units
%     ay(:,t) = Wyh*h(:,t) + bayneverlearn(:,t) + bay                      output activation
%      y(:,t) = g(ay(:,t))                                                 output units
%--------------------------------------------------------------------------
% IN   - dimIN x numT x numexamples matrix, inputs, IN(:,i,j) is an input vector at time T(i) trial j 
% f   - hidden unit nonlinearity 
% g   - output unit nonlinearity 

% ah0 - numh x numexamples matrix, initial activation of hidden units
% h0  - numh x numexamples matrix, initial values of hidden units
% bah  - numh x 1 matrix, hidden bias
% bay  - dimOUT x 1 matrix, output bias
% bahneverlearn   - numh x numT x numexamples matrix, hidden activation bias
% bhneverlearn    - numh x numT x numexamples matrix, hidden bias
% bayneverlearn - dimOUT x numT x numexamples matrix, output bias
% Whx - numh x dimIN matrix, input-to-hidden weight matrix
% Whh - numh x numh matrix, hidden-to-hidden weight matrix
% Wyh - dimOUT x numh matrix, hidden-to-output weight matrix

% ah  - hidden activation 
% h   - hidden 
% ay  - output activation
% y   - numoutputunits x numT x numexamples matrix, outputs, y(:,i,j) is the output vector at time T(i) trial j

[dimIN, numT, numexamples] = size(IN);
dimOUT = size(bay,1);
numh = size(bah,1);
permuteIN = permute(IN,[1 3 2]);% dimIN x numexamples x numT matrix, permute dimensions of array because squeeze(randn(1,5,1)) has dimensions 1 x 5 as opposed to squeeze(randn(1,1,5) which has dimensions 5 x 1
permutebahneverlearn = permute(bahneverlearn,[1 3 2]);% numh x numexamples x numT matrix, permute dimensions of array because squeeze(randn(1,5,1)) has dimensions 1 x 5 as opposed to squeeze(randn(1,1,5) which has dimensions 5 x 1
permutebhneverlearn = permute(bhneverlearn,[1 3 2]);% numh x numexamples x numT matrix, permute dimensions of array because squeeze(randn(1,5,1)) has dimensions 1 x 5 as opposed to squeeze(randn(1,1,5) which has dimensions 5 x 1
permutebayneverlearn = permute(bayneverlearn,[1 3 2]);% dimOUT x numexamples x numT matrix, permute dimensions of array because squeeze(randn(1,5,1)) has dimensions 1 x 5 as opposed to squeeze(randn(1,1,5) which has dimensions 5 x 1

% forward pass
ah = zeros(numh,numexamples,numT);% numh x numexamples x numT matrix
h = zeros(numh,numexamples,numT);% numh x numexamples x numT matrix
%ay = zeros(dimOUT,numexamples,numT);% dimOUT x numexamples x numT matrix
%y = zeros(dimOUT,numexamples,numT);% dimOUT x numexamples x numT matrix, outputs, each column of y is a desired output vector correspoding to the input vector given by the same column of IN
bahrepmat = bah*ones(1,numexamples);% numh x numexamples matrix
%bayrepmat = bay*ones(1,numexamples);% dimOUT x numexamples matrix
stuffforTau = zeros(numh,numexamples,numT);
dtoverTaurepmat = (dt./Tau)*ones(1,numexamples);% numh x numexamples matrix
ahtminus = ah0;% numh x numexamples
htminus = h0;% numh x numexamples
if isequal(bahneverlearn,0) && isequal(bhneverlearn,0) && isequal(bayneverlearn,0)
    for t=1:numT
        A = -ahtminus + Whx*permuteIN(:,:,t) + Whh*htminus + bahrepmat;% numh x numexamples
        stuffforTau(:,:,t) = A;
        ah(:,:,t) = ahtminus + dtoverTaurepmat.*A;% numh x numexamples
        h(:,:,t) = f(ah(:,:,t) ,nonlinearity{1});% numh x numexamples
        %ay(:,:,t) = Wyh*h(:,:,t) + bayrepmat;% dimOUT x numexamples
        %y(:,:,t) = f(ay(:,:,t) ,nonlinearity{2});% dimOUT x numexamples
        ahtminus = ah(:,:,t);% numh x numexamples
        htminus = h(:,:,t);% numh x numexamples
    end% for t=1:numT
    ay = pagemtimes(Wyh,h) + bay;% dimOUT x numexamples x numT matrix, pagemtimes computes Wyh*h(:,:,i) for each i, bay is broadcast
    y = f(ay ,nonlinearity{2});% dimOUT x numexamples x numT matrix
else
    for t=1:numT
        A = -ahtminus + Whx*permuteIN(:,:,t) + Whh*htminus + bahrepmat + permutebahneverlearn(:,:,t);% numh x numexamples
        stuffforTau(:,:,t) = A;
        ah(:,:,t) = ahtminus + dtoverTaurepmat.*A;% numh x numexamples
        h(:,:,t) = f(ah(:,:,t) ,nonlinearity{1}) + permutebhneverlearn(:,:,t);% numh x numexamples
        %ay(:,:,t) = Wyh*h(:,:,t) + permutebayneverlearn(:,:,t) + bayrepmat;% dimOUT x numexamples
        %y(:,:,t) = f(ay(:,:,t) ,nonlinearity{2});% dimOUT x numexamples
        ahtminus = ah(:,:,t);% numh x numexamples
        htminus = h(:,:,t);% numh x numexamples
    end% for t=1:numT
    ay = pagemtimes(Wyh,h) + permutebayneverlearn + bay;% dimOUT x numexamples x numT matrix, pagemtimes computes Wyh*h(:,:,i) for each i, bay is broadcast
    y = f(ay ,nonlinearity{2});% dimOUT x numexamples x numT matrix
end

ah = permute(ah,[1 3 2]);% numh x numT x numexamples matrix
h = permute(h,[1 3 2]);% numh x numT x numexamples matrix
ay = permute(ay,[1 3 2]);% dimOUT x numT x numexamples matrix
y = permute(y,[1 3 2]);% dimOUT x numT x numexamples matrix

end

%--------------------------------------------------------------------------
%                    compute specified nonlinearity 
%--------------------------------------------------------------------------
function F = f(IN,type,varargin)% calling the function with no specified output will only return the first output (F)
switch type
    case 'linear'
        F = IN;
    case 'logistic'
        F = 1 ./ (1 + exp(-IN));
    case 'smoothReLU'% smoothReLU or softplus 
        F = log(1 + exp(IN));% always greater than zero   
    case 'ReLU'% rectified linear units
        F = max(IN,0);
    case 'swish'% swish or SiLU (sigmoid linear unit)
        % Hendrycks and Gimpel 2016 "Gaussian Error Linear Units (GELUs)"
        % Elfwing et al. 2017 "Sigmoid-weighted linear units for neural network function approximation in reinforcement learning"
        % Ramachandran et al. 2017 "Searching for activation functions"
        sigmoid = 1./(1+exp(-IN));
        F = IN .* sigmoid;% x*sigmoid(x)
    case 'mish'% Misra 2019 "Mish: A Self Regularized Non-Monotonic Neural Activation Function
        F = IN .* tanh(log(1+exp(IN)));
    case 'GELU'% Hendrycks and Gimpel 2016 "Gaussian Error Linear Units (GELUs)"
        F = 0.5 * IN .* (1 + tanh(sqrt(2/pi)*(IN + 0.044715*IN.^3)));% fast approximating used in original paper
        %F = x.*normcdf(x,0,1);% x.*normcdf(x,0,1)  =  x*0.5.*(1 + erf(x/sqrt(2)))
        %figure; hold on; x = linspace(-5,5,100); plot(x,x.*normcdf(x,0,1),'k-'); plot(x,0.5*x.*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x.^3))),'r--')           
    case 'ELU'% exponential linear units, Clevert et al. 2015 "FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)"
        alpha = 1;
        inegativeIN = (IN < 0);
        F = IN; F(inegativeIN) = alpha * (exp(IN(inegativeIN)) - 1);     
    case 'tanh'
        F = tanh(IN);
    case 'tanhwithslope'
        a = varargin{1};
        F = tanh(a*IN);% F(x)=tanh(a*x), dFdx=a-a*(tanh(a*x).^2), d2dFdx=-2*a^2*tanh(a*x)*(1-tanh(a*x).^2)    
    case 'tanhlecun'% LeCun 1998 "Efficient Backprop" 
        F = 1.7159*tanh(2/3*IN);% F(x)=a*tanh(b*x), dFdx=a*b-a*b*(tanh(b*x).^2), d2dFdx=-2*a*b^2*tanh(b*x)*(1-tanh(b*x).^2)   
    case 'lineartanh'
        F = min(max(IN,-1),1);% -1(x<-1), x(-1<=x<=1), 1(x>1)
    case 'retanh'% rectified tanh
        F = max(tanh(IN),0);
    case 'binarymeanzero'% binary units with output values -1 and +1
        F = (IN>=0) - (IN<0);
    otherwise
        error('Unknown transfer function type');
end
end
