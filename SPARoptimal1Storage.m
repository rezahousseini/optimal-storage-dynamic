N=5;
T=1/4;
n=0:N;
t=n*T;
Iter=1;
BR=10;
filename="SPARoptimalStorageV0_0";

W=zeros(3,N+1,Iter);
x=zeros(3,N+1,Iter);
v=zeros(BR,N+1,Iter);
Rx=zeros(N+2,Iter);
R=zeros(N+2,Iter);

mkoctfile SPARoptimalValueFunctionV0_0.cc -lglpk

u=zeros(1,N);
for k= 2:N+1
	u(k)=0.9*u(k-1)+normrnd(0,0.1,1,1);
endfor
W(1,:,1)=exp(0.16+0.1*cos(2*pi.*n*T/24-5*pi/4)+u+normrnd(0,0.1,1,N+1));
W(2,:,1)=exp(0.2+0.4*cos(2*pi.*n*T/24-pi)+u+normrnd(0,0.1,1,N+1));
W(3,:,1)=exp(0.1+0.4*cos(2*pi.*n*T/24-3*pi/2)+u+normrnd(0,0.1,1,N+1));

writeOptimizationModelDataV2_1(filename,BR,...
	"g",W(1,1,1),...
	"p",W(3,1,1),...
	"r",W(2,1,1),...
	"R",R(1,1),...
	"v",v(:,1,1)...
);
[V1,xd1,xs1,xtr1]=SPARoptimalValueFunctionV0_0(filename);
[V2,xd2,xs2,xtr2]=SPARoptimalValueFunctionV0_1(W(1,1,1),W(3,1,1),W(2,1,1),R(1,1),v(:,1,1));

%for iter=1:Iter
%	u=zeros(1,N+1);
%	for k= 2:N+1
%		u(k)=0.9*u(k-1)+normrnd(0,0.1,1,1);
%	endfor
%	W(1,:,iter)=exp(0.16+0.1*cos(2*pi.*n*T/24-5*pi/4)+u+normrnd(0,0.1,1,N));
%	W(2,:,iter)=exp(0.2+0.4*cos(2*pi.*n*T/24-pi)+u+normrnd(0,0.1,1,N));
%	W(3,:,iter)=exp(0.1+0.4*cos(2*pi.*n*T/24-3*pi/2)+u+normrnd(0,0.1,1,N));

%	for k=0:N
%		R(k+2,iter)=Rx(k+1,iter);
%		[V,xd,xs,xtr]=SPARoptimalValueFunctionV0_1(W(1,k+1,iter),W(3,k+1,iter),W(2,k+1,iter),R(k+1,iter),v(:,k+1,iter));

%		Rx(k+2,iter)=R(k+2,iter)-xd+xs-xtr;

%		if(k<N)
%			for 
%		endif
%	endfor
%endfor
