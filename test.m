clear all;
close all;

numN=10;
T=1;
n=0:numN-1;
t=n*T;

numI=3;
rho=1;

S.Qmax=[10,Inf];
S.Qmin=[0,0];
S.q0=[10,Inf];
S.C=[5,30];
S.D=[5,30];
S.nul=[1,1];
S.nuc=[1,1];
S.nud=[1,1];
S.DeltaCmax=[1,30];
S.DeltaDmax=[1,30];

numS=length(S.Qmax);
numW=3;

u=zeros(1,numN);
for k=2:numN
	u(k)=0.9*u(k-1)+normrnd(0,0.1,1,1);
endfor

for k=1:numW
	g(:,k)=exp(1+1.5*cos(2*pi.*n*T/24-5*pi/4)+u+normrnd(0,0.1,1,numN));
	p(:,k)=exp(0.1+0.4*cos(2*pi.*n*T/24-3*pi/2)+u+normrnd(0,0.1,1,numN));
	r(:,k)=exp(1.17+1.6*cos(2*pi.*n*T/24-pi)+u+normrnd(0,0.1,1,numN));
endfor

P.pg=zeros(numN,numW);
P.pr=zeros(numN,numW);
P.pc=zeros(numS,numN,numW);
P.pc(1,:,:)=0.5*ones(numN,numW);
P.pc(2,:,:)=p;
P.pd=zeros(numS,numN,numW);
P.pd(1,:,:)=0.5*ones(numN,numW);
P.pd(2,:,:)=p;

%plot(t,mean(r,2),t,mean(g,2))

%mkoctfile SPARoptimalNStorage.cc -lglpk
tic;
[q,uc,ud]=SPARoptimalNStorage(rho,g,r,P,S,numI,T);
toc;

figure(1)
%plot(t,r,t,g,t,g-uc(1,:)'-uc(2,:)'-uc(3,:)'+ud(1,:)'+ud(2,:)'+ud(3,:)',t,ud(4,:)'-uc(4,:)')
plot(t,mean(r,2),t,mean(g,2),t,mean(g,2)-uc(1,:)'+ud(1,:)',t,ud(2,:)'-uc(2,:)')
legend("Nachfrage","Produktion ohne Speicher","Produktion mit Speicher","Netz")
grid on

%figure(2)
%plot(t,r-(g-uc(1,:)'+ud(1,:)'+ud(2,:)'-uc(2,:)'),t,ud(3,:)'-uc(3,:)')
%grid on

figure(3)
%plot(t,q(1,:),t,q(2,:),t,q(3,:))
plot(t,q(1,:),t,mean(g,2),t,mean(r,2))
grid on
