clear all;
close all;

numN=100;
T=1;
n=0:numN-1;
t=n*T;

numI=100;
rho=100;

S.Qmax=[4,3,1,Inf];
S.Qmin=[0,0,0,0];
S.q0=[3,2,0.5,Inf];
S.C=[0.1,0.3,0.6,10];
S.D=[0.1,0.3,0.6,10];
S.nul=[1,1,1,1];
S.nuc=[1,1,1,1];
S.nud=[1,1,1,1];

numS=length(S.Qmax);
numW=1;

u=zeros(1,numN);
for k=2:numN
	u(k)=0.9*u(k-1)+normrnd(0,0.1,1,1);
endfor

for k=1:numW
	g(:,k)=exp(0.16+0.1*cos(2*pi.*n*T/24-5*pi/4)+u+normrnd(0,0.1,1,numN));
	p(:,k)=exp(0.1+0.4*cos(2*pi.*n*T/24-3*pi/2)+u+normrnd(0,0.1,1,numN));
	r(:,k)=exp(0.17+0.4*cos(2*pi.*n*T/24-pi)+u+normrnd(0,0.1,1,numN));
endfor

P.pg=zeros(numN,numW);
P.pr=zeros(numN,numW);
P.pc=zeros(numS,numN,numW);
P.pc(4,:,:)=p;
P.pd=zeros(numS,numN,numW);
P.pd(4,:,:)=p;

mkoctfile SPARoptimalNStorage.cc -lglpk
tic;
[q,uc,ud]=SPARoptimalNStorage(rho,g,r,P,S,numI,T);
toc;

figure(1)
plot(t,r,t,g,t,g-uc(1,:)'-uc(2,:)'-uc(3,:)'+ud(1,:)'+ud(2,:)'+ud(3,:)',t,ud(4,:)'-uc(4,:)')
legend("Nachfrage","Produktion ohne Speicher","Produktion mit Speicher","Netz")
grid on

%figure(2)
%plot(t,r-(g-uc(1,:)'+ud(1,:)'+ud(2,:)'-uc(2,:)'),t,ud(3,:)'-uc(3,:)')
%grid on

figure(3)
plot(t,q(1,:),t,q(2,:),t,q(3,:))
grid on
