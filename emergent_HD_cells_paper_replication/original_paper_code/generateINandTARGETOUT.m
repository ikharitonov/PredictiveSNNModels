function [IN, TARGETOUT, itimeRNN, angle_radians, angularvelocity_store] = generateINandTARGETOUT(dimIN,dimOUT,numT,numtrials,randseed,noiseamplitude_input,angle0duration,ANGULARVELOCITY,BOUNDARY)
%close all; clear all; dimIN = 3; dimOUT = 2; numT = 500; numtrials = 1001; randseed = 1; noiseamplitude_input = 0; angle0duration = 10; ANGULARVELOCITY.angularvelocitymindegrees = -Inf; ANGULARVELOCITY.angularvelocitymaxdegrees = Inf; ANGULARVELOCITY.angularmomentum = 0.8; ANGULARVELOCITY.sd = .03; BOUNDARY.periodic = 1;% comment out function declaration and uncomment this line to run function as script
%close all; clear all; dimIN = 3; dimOUT = 2; numT = 500; numtrials = 1001; randseed = 1; noiseamplitude_input = 0; angle0duration = 10; ANGULARVELOCITY.angularvelocitymindegrees = -Inf; ANGULARVELOCITY.angularvelocitymaxdegrees = Inf; ANGULARVELOCITY.discreteangularvelocitydegrees = [-8 0 0 0 8]; BOUNDARY.periodic = 1;% comment out function declaration and uncomment this line to run function as script
%close all; clear all; dimIN = 3; dimOUT = 2; numT = 500; numtrials = 1001; randseed = 1; noiseamplitude_input = 0; angle0duration = 10; ANGULARVELOCITY.angularvelocitymindegrees = -Inf; ANGULARVELOCITY.angularvelocitymaxdegrees = Inf; ANGULARVELOCITY.angularmomentum = 0.8; ANGULARVELOCITY.sd = .03; BOUNDARY.periodic = 0; BOUNDARY.minangle = 0; BOUNDARY.maxangle = 200;% comment out function declaration and uncomment this line to run function as script

%---------------------------------------------
%                 INPUTS
%---------------------------------------------
% dimIN:      number of input units
% dimOUT:     number of output units
% numT:       number of time-steps in simulation
% numtrials:  number of trials 
% randseed:   seed for random number generator
% angle0duration = 10;% angle0 input (sin(angle0) and cos(angle0)) is nonzero for angle0duration timesteps at beginning of trial
% ANGULARVELOCITY.discreteangularvelocitydegrees = [-8 0 0 0 8];% if this variable exists then the angular velocity is drawn randomly from this discrete set at each timestep, and angularmomentum and sd are not used
% ANGULARVELOCITY.angularmomentum = 0.8;% angularvelocity(t) = sd*randn + angularmomentum * angularvelocity(t-1), angularmomentum and sd are not used when using ANGULARVELOCITY.discreteangularvelocitydegrees
% ANGULARVELOCITY.sd = .03;% at each timestep the new angularvelocity is a gaussian random variable with mean 0 and standard deviation sd, angularmomentum and sd are not used when using ANGULARVELOCITY.discreteangularvelocitydegrees
% ANGULARVELOCITY.angularvelocitymindegrees = -Inf;% minimum angular input/angular-velocity on a single timestep (degrees)
% ANGULARVELOCITY.angularvelocitymaxdegrees = Inf;% maximum angular input/angular-velocity on a single timestep (degrees)
% noiseamplitude_input = 0;
% BOUNDARY.periodic = 1;% if 1 periodic boundary conditions
% if BOUNDARY.periodic = 0 specify BOUNDARY.minangle and BOUNDARY.maxangle 


%---------------------------------------------
%                OUTPUTS
%---------------------------------------------
% IN:         dimIN x numT x numtrials matrix
% TARGETOUT:  dimOUT x numT x numtrials matrix
% itimeRNN:   dimOUT x numT x numtrials matrix, elements 0(time-point does not contribute to first term in cost function), 1(time-point contributes to first term in cost function)
% angle:      dimOUT x numT x numtrials matrix, integrated angles from input


%---------------------------------------------
%        IN, TARGETOUT, itimeRNN
%---------------------------------------------
if numtrials==0
    IN = -700*ones(dimIN,numT,numtrials);% dimIN x numT x numtrials matrix
    TARGETOUT = -700*ones(dimOUT,numT,numtrials);% dimOUT x numT x numtrials matrix
    itimeRNN = zeros(dimOUT,numT,numtrials);% dimOUT x numT x numtrials matrix, elements 0(time-point does not contribute to first term in cost function), 1(time-point contributes to first term in cost function)   
    angle_radians = zeros(dimOUT,numT,numtrials);% dimOUT x numT x numtrials matrix, integrated angles from input
    return
end
IN = zeros(dimIN,numT,numtrials);% dimIN x numT x numtrials matrix
TARGETOUT = zeros(dimOUT,numT,numtrials);% dimOUT x numT x numtrials matrix
itimeRNN = ones(dimOUT,numT,numtrials);% dimOUT x numT x numtrials matrix, elements 0(time-point does not contribute to first term in cost function), 1(time-point contributes to first term in cost function)

DISCRETEANGLES = 0; if isfield(ANGULARVELOCITY,'discreteangularvelocitydegrees'); DISCRETEANGLES = 1; end
if isfield(ANGULARVELOCITY,'discreteangularvelocitydegrees'); discreteangularvelocitydegrees = ANGULARVELOCITY.discreteangularvelocitydegrees; discreteangularvelocityradians = discreteangularvelocitydegrees*pi/180; end% if this variable exists then the angular velocity is drawn randomly from this discrete set at each timestep, and no momentum is used
if isfield(ANGULARVELOCITY,'angularmomentum'); angularmomentum = ANGULARVELOCITY.angularmomentum; end% angularvelocity(t) = sd*randn + angularmomentum * angularvelocity(t-1), angularmomentum and sd are not used when using ANGULARVELOCITY.discreteangularvelocitydegrees
if isfield(ANGULARVELOCITY,'sd'); sd = ANGULARVELOCITY.sd; end% at each timestep the new angularvelocity is a gaussian random variable with mean 0 and standard deviation sd, angularmomentum and sd are not used when using ANGULARVELOCITY.discreteangularvelocitydegrees
angularvelocitymin = ANGULARVELOCITY.angularvelocitymindegrees*pi/180;% convert angles to radians
angularvelocitymax = ANGULARVELOCITY.angularvelocitymaxdegrees*pi/180;% convert angles to radians

%---------------------------------------------
% there are 3 inputs, 
% 1) angular velocity, angle to integrate
% 2) sin(angle0)
% 3) cos(angle0)
% there are 2 outputs, sin(integrated-angle) and cos(integrated-angle)
%---------------------------------------------
rng(randseed);% http://www.walkingrandomly.com/?p=2945
if BOUNDARY.periodic==1; angle0 = 2*pi*rand(numtrials,1); end% numtrials x 1 matrix, initial angle in radians
if BOUNDARY.periodic==0; minangle = BOUNDARY.minangle*pi/180; maxangle = BOUNDARY.maxangle*pi/180; angle0 = (maxangle - minangle)*rand(numtrials,1) + minangle; end% numtrials x 1 matrix, initial angle in radians
angle = -700*ones(numT,numtrials);% integrated angularvelocity, target output of RNN (technically target output is sin(angle), cos(angle))
angularvelocity_store = -700*ones(numT,numtrials);% numT x numtrials matrix
for itrial=1:numtrials
    if itrial<=200; istart0 = randi(round(numT/3)); iend0 = min(numT,istart0 + randi(round(2*numT/3))); end% set angularvelocity for middle third of trial to 0
        
    angularvelocity = zeros(1,numT);% input to RNN
    tstart = angle0duration + 1;% start inputting nonzero angularvelocity
    angle(1:angle0duration,itrial) = angle0(itrial); 
    for t=tstart:numT% start inputting nonzero angularvelocity
        
        if itrial<=200
            if t>=istart0 && t<=iend0
                angularvelocity(t) = 0;
            else
                if DISCRETEANGLES==0; angularvelocity(t) = sd*randn + angularmomentum * angularvelocity(t-1); end
                if DISCRETEANGLES==1; angularvelocity(t) = discreteangularvelocityradians(randi(numel(discreteangularvelocityradians))); end
            end
        else
            if DISCRETEANGLES==0; angularvelocity(t) = sd*randn + angularmomentum * angularvelocity(t-1); end
            if DISCRETEANGLES==1; angularvelocity(t) = discreteangularvelocityradians(randi(numel(discreteangularvelocityradians))); end
        end
        angularvelocity(t) = min(angularvelocity(t),angularvelocitymax);% clip angularvelocity so it does not exceed angularvelocitymax and angularvelocitymin
        angularvelocity(t) = max(angularvelocity(t),angularvelocitymin);% clip angularvelocity so it does not exceed angularvelocitymax and angularvelocitymin
        anglenew = angle(t-1,itrial) + angularvelocity(t);
        if BOUNDARY.periodic==1
            angle(t,itrial) = mod(anglenew,2*pi);% integrated angularvelocity, target output of RNN (technically target output is sin(angle), cos(angle))
        end
        if BOUNDARY.periodic==0% angle cannot go outside fixed boundaries
            OUTSIDEBOUNDARY = (anglenew < minangle) || (maxangle < anglenew);
            count = 1;
            while OUTSIDEBOUNDARY
                if DISCRETEANGLES==0; angularvelocity(t) = sd*randn + angularmomentum * angularvelocity(t-1); end
                if DISCRETEANGLES==1; angularvelocity(t) = discreteangularvelocityradians(randi(numel(discreteangularvelocityradians))); end
                angularvelocity(t) = min(angularvelocity(t),angularvelocitymax);% clip angularvelocity so it does not exceed angularvelocitymax and angularvelocitymin
                angularvelocity(t) = max(angularvelocity(t),angularvelocitymin);% clip angularvelocity so it does not exceed angularvelocitymax and angularvelocitymin
                anglenew = angle(t-1,itrial) + angularvelocity(t);
                OUTSIDEBOUNDARY = (anglenew < minangle) || (maxangle < anglenew);
                count = count + 1;
            end
            angle(t,itrial) = mod(anglenew,2*pi);% integrated angularvelocity, target output of RNN (technically target output is sin(angle), cos(angle))
        end
    end
    IN(1,tstart:numT,itrial) = angularvelocity(tstart:numT);% 1 x something matrix, angle in radians
    IN(2,1:angle0duration,itrial) = sin(angle0(itrial));
    IN(3,1:angle0duration,itrial) = cos(angle0(itrial));
    angularvelocity_store(:,itrial) = angularvelocity;
end% for itrial=1:numtrials

TARGETOUT(1,:,:) = sin(angle);% numT x numtrials matrix
TARGETOUT(2,:,:) = cos(angle);% numT x numtrials matrix
itimeRNN(:,1:angle0duration,:) = 0;% dimOUT x numT x numtrials matrix, elements 0(time-point does not contribute to first term in cost function), 1(time-point contributes to first term in cost function)

angle_radians = angle;% numT x numtrials matrix, stored/integrated angle
angle_degrees = angle_radians * 180/pi;% numT x numtrials matrix


%--------------------------------------------------------------------------
%                       add noise to IN 
%--------------------------------------------------------------------------
IN = IN + noiseamplitude_input*randn(dimIN,numT,numtrials);


% figure;% input and target output 
% for itrial = [1 100 101:numtrials]
%     clf; hold on; fontsize = 15; set(gcf,'DefaultLineLineWidth',2,'DefaultLineMarkerSize',10,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)% only need to set DefaulttextFontSize if using function text
%     T = [1:numT];
%     plot([1:numT],IN(1,:,itrial),'k-');% for legend
%     plot(T(itimeRNN(1,:,itrial)==1),TARGETOUT(1,itimeRNN(1,:,itrial)==1,itrial),'r--')% for legend
%     
%     plot([1:numT],IN(:,:,itrial),'k-')
%     for i=1:dimOUT
%         plot(T(itimeRNN(i,:,itrial)==1),TARGETOUT(i,itimeRNN(i,:,itrial)==1,itrial),'r--');
%     end
%     xlabel('Time steps')
%     legend('Input','Target output','location','best');
%     title(sprintf('Trial %g, %g time-steps in simulation',itrial,numT ));
%     set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
%     axis tight; axis([-Inf Inf min(ylim)-abs(max(ylim)-min(ylim))/100  max(ylim)+abs(max(ylim)-min(ylim))/100])
%     pause
% end
% % 


% figure;% input and target output, with integrated-angle in degrees
% for itrial = [1:numtrials]
%     clf; hold on; fontsize = 15; set(gcf,'DefaultLineLineWidth',2,'DefaultLineMarkerSize',10,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)% only need to set DefaulttextFontSize if using function text
%     T = [1:numT];
%     plot([1:numT],IN(1,:,itrial),'k-');% for legend
%     plot(T(itimeRNN(1,:,itrial)==1),TARGETOUT(1,itimeRNN(1,:,itrial)==1,itrial),'r-')% for legend
%     
%     plot([1:numT],IN(:,:,itrial),'k-')
%     for i=1:dimOUT
%         plot(T(itimeRNN(i,:,itrial)==1),TARGETOUT(i,itimeRNN(i,:,itrial)==1,itrial),'r-');
%     end
%     
%     plot(T,angle_degrees(:,itrial),'g-')% integrated angle in degrees
%     xlabel('Time steps')
%     legend('Input','Target output','location','best');
%     title(sprintf('Trial %g, %g time-steps in simulation',itrial,numT ));
%     set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
%     axis tight; axis([-Inf Inf min(ylim)-abs(max(ylim)-min(ylim))/100  max(ylim)+abs(max(ylim)-min(ylim))/100])
%     ylim([0 360])
%     pause
% end


%%
%distribution of angular inputs
% rng(2)
% A = full(sprand(numtrials,numT,0.5)) * (inputmax - inputmin) + inputmin; A = A*180/pi;
% A = rand(numtrials,numT) * (inputmax - inputmin) + inputmin; A = A*180/pi;

% A = IN(1,:,:) * 180/pi; A = A(:);% angular inputs in degrees
% A = round(10*A)/10;% round angular inputs to nearest 0.1 degrees
% %A(A==0) = [];% most common element is 0, so remove this to see what rest of distribution looks like
% % find the unique elements of a vector x and how many times each unique element of x occurs
% x = A;
% x = sort(x(:)); 
% difference = diff([x;max(x)+1]); 
% uniqueelements = x(difference~=0);
% count = diff(find([1;difference]));
% figure% plot histogram
% clf; hold on; fontsize = 12; set(gcf,'DefaultLineLineWidth',4,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)  
% bar(uniqueelements,count,'BarWidth',0.8,'FaceColor','b','EdgeColor','b')
% ylabel('Frequency')
% axis tight


