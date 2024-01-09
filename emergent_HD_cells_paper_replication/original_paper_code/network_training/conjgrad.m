function [xstore, istore] = conjgrad( Afunc, b, x0, maxiters, miniters, Mdiag )
% Shewchuk 1994 "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain"
% preconditioned conjugate gradient algorithm to minimize f(x) = 1/2*x'*A*x - b'*x
% d(f(x))/dx = 1/2*(A+A')*x - b 
%              = A*x - b if A = A'

%--------------------------------------------------------------------------
%                              test conjgrad
%          comment out line 1 and run as script, solution should equal xstore(:,end)
%--------------------------------------------------------------------------
% close all; clear all;
% A = [17 2; 2 7];  b = [2; 2];% solution = [0.0870; 0.2609]
% A = [3 2; 2 6]; b = [2; -8];% solution = [2; -2]
% x0 = [0; 0];
% solution = A\b;% inv(A)*b;
% [X,Y] = meshgrid(-1:.2:3,-3:.2:1);
% Z = 0.5*A(1,1)*X.*X + A(1,2)*X.*Y + 0.5*A(2,2)*Y.*Y - b(1)*X - b(2)*Y;% 1/2*x'*A*x - b'*x
% figure; hold on; 
% contour(X,Y,Z,30); plot(solution(1),solution(2),'ko')
% figure; hold on;
% levels = [linspace(min(min(Z))+0.1,max(max(Z)),20)]; [C h] = contour(X,Y,Z,levels,'k-');
% set(h,'showtext','on')
% handle = clabel(C,h,'fontsize',12);
% for a=1:length(handle)
%     s = get(handle(a),'String');% get string
%     s = str2num(s);% convert it to number
%     s = sprintf('%.3g',s);% display three digits 
%     set(handle(a),'String',s);% place it back in the figure
% end
% plot(solution(1),solution(2),'ko')
% axis image; axis tight; title(sprintf('Contours of 1/2*x''*A*x - b''*x, minimum is %.3g',min(min(Z))))
% maxiters = 2; miniters = 1; Mdiag = eye(size(A));% parameters for function
%--------------------------------------------------------------------------


phi = zeros(maxiters,1);
inext = 5; 
%inext = 1;% if running as a script
imult = 1.3;

istore = [];
xstore = [];

r = b - Afunc(x0);
%r = b - A*x0;% if running as a script
d = Mdiag\r;% inv(Mdiag)*r;
delta_new = r'*d;
x = x0;
for i = 1:maxiters
    Ad = Afunc(d);% compute the matrix-vector product.  This is where 95% of the work in HF lies  
    %Ad = A*d;% if running as a script
    dAd = d'*Ad;
    % The Gauss-Newton matrix should never have negative curvature.  The Hessian easily could unless your objective is convex
    if dAd <= 0; disp('Negative Curvature!'); break; end
    
    alpha = delta_new/dAd;
    x = x + alpha*d;
    r = r - alpha*Ad; 
    s = Mdiag\r;
    delta_old = delta_new;
    delta_new = r'*s;
    beta = delta_new/delta_old;
    d = s + beta*d; 
    
    if i == ceil(inext)
        istore(end+1) = i;% store i
        xstore(:,end+1) = x;% store x
        inext = inext*imult;
    end    
    % Martens 2010 "Deep learning via Hessian-free optimization"
    %the stopping criterion here becomes largely unimportant once you
    %optimize your function past a certain point, as it will almost never
    %kick in before you reach i = maxiters.  And if the value of maxiters
    %is set so high that this never occurs, you probably have set it too
    %high
    k = max(10,ceil(0.1*i));
    phi(i) = 0.5*double((-b-r)'*x);% phi = 1/2*x'*A*x - b'*x, r = b - A*x
    if i > k && phi(i-k) < 0 && (phi(i) - phi(i-k))/phi(i) < 0.0005*k && i >= miniters
        break;
    end   
end% for i=1:maxiters

if i ~= ceil(inext)
    istore(end+1) = i;
    xstore(:,end+1) = x;
end
