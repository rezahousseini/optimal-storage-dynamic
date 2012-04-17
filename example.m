clear all;
close all;

numN=10;
T=1/2;
n=0:numN-1;
t=n*T;

numI=10;
rho=100;

S.Qmax=[10,Inf];
S.Qmin=[0,0];
S.q0=[10,Inf];
S.C=[5,30];
S.D=[5,30];
S.etal=[1,1];
S.etac=[1,1];
S.etad=[1,1];
S.DeltaCmax=[5,30];
S.DeltaDmax=[5,30];

numS=length(S.Qmax);
numW=3;

u=zeros(1,numN);
for k=2:numN
	u(k)=0.9*u(k-1)+normrnd(0,0.1,1,1);
end


for k=1:numW
	g(:,k)=exp(1+1.5*cos(2*pi.*n*T/24-5*pi/4));%+u+normrnd(0,0.1,1,numN));
	p(:,k)=exp(0.8+0.4*cos(2*pi.*n*T/24-3*pi/2));%+u+normrnd(0,0.1,1,numN));
	r(:,k)=exp(1.1+1.4*cos(2*pi.*n*T/24-pi));%+u+normrnd(0,0.1,1,numN));
end

P.pg=zeros(numN,numW);
P.pr=zeros(numN,numW);
P.pc=zeros(numS,numN,numW);
P.pc(1,:,:)=0.2*ones(numN,numW);
P.pc(2,:,:)=p;
P.pd=zeros(numS,numN,numW);
P.pd(1,:,:)=0.2*ones(numN,numW);
P.pd(2,:,:)=p;

Parm.gamma=0.5;
Parm.alpha0=1;
Parm.deltaStepMult=0.8;
Parm.a=1;
Parm.b=1000;
Parm.c=1;
Parm.epsilon=1;

tic;
[q,uc,ud,cost,costIter]=ADPoptimalNStorage(rho,g,r,P,S,numI,T,Parm);
toc;

figure(1)
plot(t,mean(r,2),t,mean(g,2),t,mean(g,2)-uc(1,:)'+ud(1,:)',t,ud(2,:)'-uc(2,:)',t,ud(1,:)'-uc(1,:)')
legend('Nachfrage','Produktion ohne Speicher','Produktion mit Speicher','Netz','Speicher')
grid on

figure(3)
plot(t,q(1,:),t,mean(g,2),t,mean(r,2),t,p)
grid on

figure(4)
plot(cumsum(cost))
grid on
