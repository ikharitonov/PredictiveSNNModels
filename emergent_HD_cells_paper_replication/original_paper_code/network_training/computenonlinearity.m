function [F, dFdx, d2Fdx2] = computenonlinearity(x,nonlinearity,varargin)% calling the function with no specified output will only return the first output, nonlinearity can be 'linear' 'logistic' 'tanh' 'lineartanh' 'retanh' 'ReLU' 'ELU' 'swish' 'mish' 'GELU'
    if nargout==1; F = f(x,nonlinearity,varargin); end
    if nargout==2; [F, dFdx] = f(x,nonlinearity,varargin); end
    if nargout==3; [F, dFdx, d2Fdx2] = f(x,nonlinearity,varargin); end
end

%--------------------------------------------------------------------------
%          compute specified nonlinearity and its derivative
%--------------------------------------------------------------------------
function [F, dFdx, d2Fdx2] = f(x,type,varargin)% calling the function with no specified output will only return the first output (F)
%function [F, dFdx, d2Fdx2, d3Fdx3] = f(x,type)% calling the function with no specified output will only return the first output (F)
switch type
    case 'linear'
        F = x;
        if nargout > 1
            dFdx = ones(size(x)); 
            d2Fdx2 = zeros(size(F)); 
        end
    case 'logistic'
        F = 1 ./ (1 + exp(-x));
        if nargout > 1
            dFdx = F - F.^2;% dFdx = F(x)-F(x).^2 
            d2Fdx2 = F.*(1-F).*(1-2*F);
            %d3Fdx3 = d2Fdx2 .* (1-2*F) - 2*dFdx.^2;
        end
    case 'smoothReLU'% smoothReLU or softplus
        F = log(1 + exp(x));% always greater than zero
        if nargout > 1
            dFdx = 1 ./ (1 + exp(-x));% dFdx = 1./(1 + exp(-x)) = 1 - exp(-F)
            d2Fdx2 = dFdx - dFdx.^2;
        end 
    case 'ReLU'% rectified linear units
        F = max(x,0);
        if nargout > 1
            dFdx = (x > 0); 
            d2Fdx2 = zeros(size(F));
        end
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
        if nargout > 1
            dFdx = ones(size(F)); dFdx(inegativex) = F(inegativex) + alpha;
            d2Fdx2 = zeros(size(F)); d2Fdx2(inegativex) = dFdx(inegativex);
        end        
    case 'tanh'
        F = tanh(x);
        if nargout > 1
            dFdx = 1 - F.^2;% dFdx = 1-F(x).^2
            d2Fdx2 = -2*F.*(1-F.^2);
        end
    case 'tanhwithslope'
        a = varargin{1}{1};
        F = tanh(a*x);
        if nargout > 1
            dFdx = a - a*(F.^2);
            d2Fdx2 = -2*a^2*F.*(1-F.^2);% F(x)=tanh(a*x), dFdx=a-a*(tanh(a*x).^2), d2dFdx=-2*a^2*tanh(a*x)*(1-tanh(a*x).^2) 
        end     
    case 'tanhlecun'% LeCun 1998 "Efficient Backprop" 
        F = 1.7159*tanh(2/3*x);% F(x)=a*tanh(b*x), dFdx=a*b-a*b*(tanh(b*x).^2), d2dFdx=-2*a*b^2*tanh(b*x)*(1-tanh(b*x).^2)
        if nargout > 1
            dFdx = 1.7159*2/3 - 2/3*(F.^2)/1.7159;
            d2Fdx2 = -2*(2/3)^2*F.*(1-(F.^2)/(1.7159^2));
        end 
    case 'scaledtanh'
        a = varargin{1}{1}; b = varargin{1}{2};
        F = a.*tanh(b.*x);% F(x)=a*tanh(b*x), dFdx=a*b-a*b*(tanh(b*x).^2), d2dFdx=-2*a*b^2*tanh(b*x)*(1-tanh(b*x).^2)
        if nargout > 1
            dFdx = a.*b - b.*(F.^2)./a;
            d2Fdx2 = -2*(b.^2).*F.*(1-(F.^2)./(a.^2));
        end       
    case 'lineartanh'
        F = min(max(x,-1),1);% -1(x<-1), x(-1<=x<=1), 1(x>1)
        if nargout > 1
            dFdx = ((x>-1) .* (x<1)) + 0.5*(x==-1) + 0.5*(x==1);% 0(x<-1), 1(-1<=x<=1), 0(x>1) 
            %dFdx = ((F>-1) .* (F<1));% 0(F<=-1), 1(-1<F<1), 0(F>=1), not quite right at x=-1 and x=1
            d2Fdx2 = zeros(size(F));
        end
    case 'retanh'
        F = max(tanh(x),0);
        if nargout > 1 
            dFdx = (1 - F.^2) .* (x > 0);% dFdx = 1-F(x).^2 
            d2Fdx2 = -2*F.*(1-F.^2) .* (x > 0);
        end
    case 'penalizedtanh'% Xu et al. 2016 "Revise saturated activation functions"
        a = 0.25;
        F = max(a*tanh(x),tanh(x));% tanh(x) if x>0, a*tanh(x) otherwise
        if nargout > 1
            dFdx = (a - (F.^2)/a) .* (x<0) + (1 - (F.^2)) .* (x>=0);% figure; hold on; a = 0.25; x = linspace(-2,2,100); plot(x,1-tanh(x).^2,'k-'); plot(x,a-a*(tanh(x).^2),'r-'); legend('d(tanh(x))/dx','d(a*tanh(x))/dx')
            d2Fdx2 = (-2*F.*(1-(F.^2)/(a^2))) .* (x<0) + (-2*F.*(1-(F.^2))) .* (x>=0);
        end    
    case 'binarymeanzero'% binary units with output values -1 and +1
        F = (x>=0) - (x<0);
        if nargout > 1
            dFdx = zeros(size(F));
            d2Fdx2 = zeros(size(F));
        end
    otherwise
        error('Unknown transfer function type');
end
end







