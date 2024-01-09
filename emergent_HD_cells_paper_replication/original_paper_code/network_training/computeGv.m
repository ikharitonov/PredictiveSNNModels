function Gv = computeGv(v,model)% CJ Cueva 7.22.2017
%function [Gv, Rah, Rh] = computeGv(v,model)
% LEARNPARAMETERS_W:  if 1 learn W, if 0 treat as constant
% L2REGULARIZE_W:  if 1 regularize W with lambdaL2, larger lambdaL2 = more regularization = smaller parameters
% permuteIN: dimIN x numexamples x numT matrix
% permuteh:   numh x numexamples x numT matrix
% permuteOUT: dimOUT x numexamples x numT matrix
% permuteitimeRNN: dimOUT x numexamples x numT matrix
LEARNPARAMETERS_Whx = model.LEARNPARAMETERS_Whx;
LEARNPARAMETERS_Whh = model.LEARNPARAMETERS_Whh;
LEARNPARAMETERS_Wyh = model.LEARNPARAMETERS_Wyh;
LEARNPARAMETERS_bah = model.LEARNPARAMETERS_bah;
LEARNPARAMETERS_bay = model.LEARNPARAMETERS_bay;
LEARNPARAMETERS_Tau = model.LEARNPARAMETERS_Tau;
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
permuteIN = model.permuteIN;
permuteah = model.permuteah;
permuteh = model.permuteh;
permuteh_withoutbias = model.permuteh_withoutbias;
permutey = model.permutey;
permuteitimeRNN = model.permuteitimeRNN;
nonlinearity = model.nonlinearity;
dt = model.dt;
lambdaL2 = model.lambdaL2;% L2 regularization on parameters
lambdahL2 = model.lambdahL2;% L2 regularization on h
lambdaSD = model.lambdaSD;
stuffforTau = model.stuffforTau;

numh = size(bah,1);% number of hidden units
dimOUT = size(bay,1);% number of outputs
[dimIN, numexamples, numT] = size(permuteIN);% dimIN x numexamples x numT matrix, permute dimensions of array because squeeze(randn(1,5,1)) has dimensions 1 x 5 as opposed to squeeze(randn(1,1,5) which has dimensions 5 x 1
numelementstoregularize_L2 = 0;% L2 regularization on parameters
if L2REGULARIZE_Whx==1 && LEARNPARAMETERS_Whx==1; numelementstoregularize_L2 = numelementstoregularize_L2 + numel(Whx); end 
if L2REGULARIZE_Whh==1 && LEARNPARAMETERS_Whh==1; numelementstoregularize_L2 = numelementstoregularize_L2 + numel(Whh); end 
if L2REGULARIZE_Wyh==1 && LEARNPARAMETERS_Wyh==1; numelementstoregularize_L2 = numelementstoregularize_L2 + numel(Wyh); end 
if L2REGULARIZE_bah==1 && LEARNPARAMETERS_bah==1; numelementstoregularize_L2 = numelementstoregularize_L2 + numel(bah); end 
if L2REGULARIZE_bay==1 && LEARNPARAMETERS_bay==1; numelementstoregularize_L2 = numelementstoregularize_L2 + numel(bay); end 
if L2REGULARIZE_Tau==1 && LEARNPARAMETERS_Tau==1; numelementstoregularize_L2 = numelementstoregularize_L2 + numel(Tau); end 


% order of parameters
% Whx, Whh, Wyh, bah, bay, Tau
vWhx = zeros(numh,dimIN);
vWhh = zeros(numh,numh);
vWyh = zeros(dimOUT,numh);
vbah = zeros(numh,1);
vbay = zeros(dimOUT,1);
vTau = zeros(numh,1);
% parameters that can be learned depending on LEARNPARAMETERS - Whx, Whh, Wyh, bah, bay, Tau
% For example, if LEARNPARAMETERS_Whx==1 overwrite Whx passed in explicitly to function with the values in the vector parameters 
ii = 1;
if LEARNPARAMETERS_Whx==1; vWhx(:) = v(1:1+numh*dimIN-1); ii = 1 + numh*dimIN; end
if LEARNPARAMETERS_Whh==1; vWhh(:) = v(ii:ii+numh*numh-1); ii = ii + numh*numh; end
if LEARNPARAMETERS_Wyh==1; vWyh(:) = v(ii:ii+numh*dimOUT-1); ii = ii + dimOUT*numh; end
if LEARNPARAMETERS_bah==1; vbah(:) = v(ii:ii+numh-1); ii = ii + numh; end
if LEARNPARAMETERS_bay==1; vbay(:) = v(ii:ii+dimOUT-1); ii = ii + dimOUT; end
if LEARNPARAMETERS_Tau==1; vTau(:) = v(ii:ii+numh-1); ii = ii + numh; end

permutebahneverlearn = permute(bahneverlearn,[1 3 2]);% numh x numexamples x numT matrix, permute dimensions of array because squeeze(randn(1,5,1)) has dimensions 1 x 5 as opposed to squeeze(randn(1,1,5) which has dimensions 5 x 1
ah = permuteah;% numh x numexamples x numT matrix, originally ah is a numh x numT x numexamples matrix
h = permuteh;% numh x numexamples x numT matrix, originally h is a numh x numT x numexamples matrix
h_withoutbias = permuteh_withoutbias;% numh x numexamples x numT matrix, originally h is a numh x numT x numexamples matrix
y = permutey;% dimOUT x numexamples x numT matrix, originally y is a dimOUT x numT x numexamples matrix
dyda = df(y,nonlinearity{2});% dimOUT x numexamples x numT matrix
%[dhda, d2hda2] = df(h,nonlinearity{1});% numh x numexamples x numT matrix
dhda = df(h_withoutbias,nonlinearity{1});% numh x numexamples x numT matrix

% forward pass 
if isfield(model,'singleprecision') 
    bahrepmat = bah*ones(1,numexamples,'single');% numh x numexamples matrix
    Rah = zeros(numh,numexamples,numT,'single');
    Rh = zeros(numh,numexamples,numT,'single');
    vbahrepmat = vbah*ones(1,numexamples,'single');% numh x numexamples
    Rahtminus = zeros(numh,numexamples,'single');% numh x numexamples
    Rhtminus = zeros(numh,numexamples,'single');% numh x numexamples
else
    bahrepmat = bah*ones(1,numexamples);% numh x numexamples matrix
    Rah = zeros(numh,numexamples,numT);
    Rh = zeros(numh,numexamples,numT);
    vbahrepmat = vbah*ones(1,numexamples);% numh x numexamples
    Rahtminus = zeros(numh,numexamples);% numh x numexamples
    Rhtminus = zeros(numh,numexamples);% numh x numexamples
end
dtoverTaurepmat = (dt./Tau)*ones(1,numexamples);% numh x numexamples matrix
A = (-vTau*dt./(Tau.^2)) * ones(1,numexamples);% numh x numexample matrix
ahtminus = ah0;% numh x numexamples
htminus = h0;% numh x numexamples
if isequal(bahneverlearn,0)
    for t=1:numT
        Rah(:,:,t) = Rahtminus +                    A .* (-ahtminus + Whx*permuteIN(:,:,t) + Whh*htminus + bahrepmat) +...% numh x numexamples matrix
                                    dtoverTaurepmat .* (-Rahtminus + vWhx*permuteIN(:,:,t) + vWhh*htminus + Whh*Rhtminus + vbahrepmat);% numh x numexamples matrix                       
        Rh(:,:,t) = dhda(:,:,t).*Rah(:,:,t);% numh x numexamples matrix
        Rahtminus = Rah(:,:,t);% numh x numexamples
        Rhtminus = Rh(:,:,t);% numh x numexamples
        ahtminus = ah(:,:,t);% numh x numexamples
        htminus = h(:,:,t);% numh x numexamples
    end% for t=1:numT
else
    for t=1:numT
        Rah(:,:,t) = Rahtminus +                    A .* (-ahtminus + Whx*permuteIN(:,:,t) + Whh*htminus + bahrepmat + permutebahneverlearn(:,:,t)) +...% numh x numexamples matrix
                                    dtoverTaurepmat .* (-Rahtminus + vWhx*permuteIN(:,:,t) + vWhh*htminus + Whh*Rhtminus + vbahrepmat);% numh x numexamples matrix                      
        Rh(:,:,t) = dhda(:,:,t).*Rah(:,:,t);% numh x numexamples matrix
        Rahtminus = Rah(:,:,t);% numh x numexamples
        Rhtminus = Rh(:,:,t);% numh x numexamples
        ahtminus = ah(:,:,t);% numh x numexamples
        htminus = h(:,:,t);% numh x numexamples
    end% for t=1:numT
end
if isfield(model,'singleprecision') 
    Ray = zeros(dimOUT,numexamples,numT,'single');
    Ray(:,:) = vWyh*h(:,:) + Wyh*Rh(:,:) + vbay*ones(1,numT*numexamples,'single');% dimOUT x numT*numexamples matrix
else
    Ray = zeros(dimOUT,numexamples,numT);
    Ray(:,:) = vWyh*h(:,:) + Wyh*Rh(:,:) + vbay*ones(1,numT*numexamples);% dimOUT x numT*numexamples matrix
end


% backward pass
sumitimeRNN = sum(permuteitimeRNN(:)==1);
if sumitimeRNN==0; sumitimeRNN = 1; end% dividing by 0 creates NaN so divide by 1 instead, this line would not be a good idea if sumitimeRNN was used to multiply other numbers because multiplying by 0 and 1 are very different!
deltay = dyda .* Ray;
deltay(permuteitimeRNN==0) = 0;
delta = dhda .* reshape(Wyh'*deltay(:,:),numh,numexamples,numT)/sumitimeRNN  +... 
        (lambdahL2/(numexamples*numT*numh)) * 2*Rh .* dhda +...
        (lambdaSD/(numexamples*numT*numh)) * (dhda .* Rah);% numh x numexamples x numT matrix, deltah + deltahL1 + deltahL2 + deltaSD, CHANGE FROM THE BACKPROPAGATION ALGORITHM
for t=numT-1:-1:1
    delta(:,:,t)    = delta(:,:,t)    + (1-dtoverTaurepmat).*delta(:,:,t+1)    + dhda(:,:,t) .* ( (((dt./Tau)*ones(1,numh)) .* Whh)'*delta(:,:,t+1));% numh x numexamples matrix
end


% compute gradients
GvWhx = []; GvWhh = []; GvWyh = []; Gvbah = []; Gvbay = []; GvTau = [];
A = delta(:,:);% numh x numexamples*numT matrix
if LEARNPARAMETERS_Whx==1; GvWhx = ((dt./Tau)*ones(1,dimIN)) .* (A * permuteIN(:,:)'); end% numh x dimIN matrix
if LEARNPARAMETERS_Whh==1; GvWhh = ((dt./Tau)*ones(1,numh))  .* (A * [h0 reshape(h(:,:,1:numT-1),numh,numexamples*(numT-1))]'); end% numh x numh matrix
if LEARNPARAMETERS_Wyh==1; GvWyh = deltay(:,:) * h(:,:)' / sumitimeRNN; end% dimOUT x numh matrix
if LEARNPARAMETERS_bah==1; Gvbah = (dt./Tau) .* sum(A,2); end% numh x 1 matrix
if LEARNPARAMETERS_bay==1; Gvbay = sum(deltay(:,:),2) / sumitimeRNN; end% dimOUT x 1 matrix
if LEARNPARAMETERS_Tau==1; GvTau = (-dt./(Tau.^2)) .* sum(A.*stuffforTau(:,:),2); end% numh x 1 matrix
if L2REGULARIZE_Whx==1 && LEARNPARAMETERS_Whx==1; GvWhx = GvWhx + (2*lambdaL2/numelementstoregularize_L2)*vWhx; end 
if L2REGULARIZE_Whh==1 && LEARNPARAMETERS_Whh==1; GvWhh = GvWhh + (2*lambdaL2/numelementstoregularize_L2)*vWhh; end 
if L2REGULARIZE_Wyh==1 && LEARNPARAMETERS_Wyh==1; GvWyh = GvWyh + (2*lambdaL2/numelementstoregularize_L2)*vWyh; end 
if L2REGULARIZE_bah==1 && LEARNPARAMETERS_bah==1; Gvbah = Gvbah + (2*lambdaL2/numelementstoregularize_L2)*vbah; end 
if L2REGULARIZE_bay==1 && LEARNPARAMETERS_bay==1; Gvbay = Gvbay + (2*lambdaL2/numelementstoregularize_L2)*vbay; end 
if L2REGULARIZE_Tau==1 && LEARNPARAMETERS_Tau==1; GvTau = GvTau + (2*lambdaL2/numelementstoregularize_L2)*vTau; end 
Gv = [GvWhx(:); GvWhh(:); GvWyh(:); Gvbah; Gvbay; GvTau];% numparameters x 1 matrix


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

function [dFdx, d2Fdx2] = df(F,type,varargin)% input has already been passed through nonlinearity, F = f(x), % calling the function with no specified output will only return the first output (dFdx)
%function [dFdx, d2Fdx2, d3Fdx3] = df(F,type)% input has already been passed through nonlinearity, F = f(x), % calling the function with no specified output will only return the first output (dFdx)
switch type
    case 'linear'
        dFdx = ones(size(F));
        if nargout > 1; d2Fdx2 = zeros(size(F)); end
    case 'logistic'
        dFdx = F - F.^2;% dfdx = f(x)-f(x).^2 = F-F.^2
        if nargout > 1; d2Fdx2 = F.*(1-F).*(1-2*F); end
        %if nargout > 1; d2Fdx2 = F.*(1-F).*(1-2*F); d3Fdx3 = d2Fdx2 .* (1-2*F) - 2*dFdx.^2; end
    case 'smoothReLU'% smoothReLU or softplus
        dFdx = 1 - exp(-F);% dFdx = 1./(1 + exp(-x)) = 1 - exp(-F)
        if nargout > 1; d2Fdx2 = dFdx - dFdx.^2; end
    case 'ReLU'% rectified linear units
        dFdx = (F > 0);
        if nargout > 1; d2Fdx2 = zeros(size(F)); end
    case 'ELU'% exponential linear units, Clevert et al. 2015 "FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)"
        alpha = 1;
        inegativex = (F < 0);% F < 0 is the same as x < 0 for ELU nonlinearity
        dFdx = ones(size(F)); dFdx(inegativex) = F(inegativex) + alpha;
        if nargout > 1; d2Fdx2 = zeros(size(F)); d2Fdx2(inegativex) = dFdx(inegativex); end
    case 'tanh'
        dFdx = 1 - F.^2;% dfdx = 1-f(x).^2 = 1-F.^2
        if nargout > 1; d2Fdx2 = -2*F.*(1-F.^2); end
    case 'tanhwithslope'
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



