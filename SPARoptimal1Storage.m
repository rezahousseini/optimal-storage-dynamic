N=100;
T=1/4;
n=0:N-1;
t=n*T;
Iter=10;
BR=100;
BW=4;

% Storage parameters
S.Qmax=1;
S.q0=S.Qmax;
S.C=0.5;
S.D=0.5;
S.nul=0.99;
S.nuc=0.9;
S.nud=0.9;

rho=S.Qmax*BR; % 1/nu

u=zeros(1,N);
for k= 2:N
	u(k)=0.9*u(k-1)+normrnd(0,0.1,1,1);
endfor

W.g=zeros(BW,N);
W.p=zeros(BW,N);
W.r=zeros(BW,N);

% Create set W
for m=1:BW
	W.g(m,:)=exp(0.16+0.1*cos(2*pi.*n*T/24-5*pi/4)+u+normrnd(0,0.1,1,N));
	W.p(m,:)=exp(0.1+0.4*cos(2*pi.*n*T/24-3*pi/2)+u+normrnd(0,0.1,1,N));
	W.r(m,:)=exp(0.2+0.4*cos(2*pi.*n*T/24-pi)+u+normrnd(0,0.1,1,N));
endfor

xd=zeros(1,N);
xs=zeros(1,N);
xi=zeros(1,N);
xe=zeros(1,N);
v=zeros(BW*N,BR+1);
Rx=zeros(1,N);
R=zeros(1,N);
NV=zeros(BW*N,BR+1); % Number of visits to the corresponding state
R0=round(S.q0*rho); % Storage level initialization

mkoctfile SPARoptimalValueFunctionV0_1.cc -lglpk

for iter=1:Iter
	% Generate sample
	sample=randi(BW,1,N);

	for k=1:N
		% Compute pre-decision asset level
		if(k == 1)
			R(k)=R0;
		else
			R(k)=Rx(k-1);
		endif

		% Find optimal value function and compute post-decision asset level
		[V,Rx(k)]=SPARoptimalValueFunctionV0_1(rho,W.g(sample(k),k),W.p(sample(k),k),W.r(sample(k),k),R(k),v((sample(k)-1)*N+k,:),T,S);

		% Count number of visits
		NV((sample(k)-1)*N+k,Rx(k))=NV((sample(k)-1)*N+k,Rx(k))+1;

		if(k<N)
			% Observe sample slopes
			[V1,Rx1]=SPARoptimalValueFunctionV0_1(rho,W.g(sample(k+1),k+1),W.p(sample(k+1),k+1),W.r(sample(k+1),k+1),Rx(k),v((sample(k+1)-1)*N+(k+1),:),T,S);
			if(Rx(k) == 1)
				V2=0;
			else
				[V2,Rx2]=SPARoptimalValueFunctionV0_1(rho,W.g(sample(k+1),k+1),W.p(sample(k+1),k+1),W.r(sample(k+1),k+1),Rx(k)-1,v((sample(k+1)-1)*N+(k+1),:),T,S);
			endif
			vhatlo=V1-V2;
			[V1,Rx1]=SPARoptimalValueFunctionV0_1(rho,W.g(sample(k+1),k+1),W.p(sample(k+1),k+1),W.r(sample(k+1),k+1),Rx(k)+1,v((sample(k+1)-1)*N+(k+1),:),T,S);
			[V2,Rx2]=SPARoptimalValueFunctionV0_1(rho,W.g(sample(k+1),k+1),W.p(sample(k+1),k+1),W.r(sample(k+1),k+1),Rx(k),v((sample(k+1)-1)*N+(k+1),:),T,S);
			vhatup=V1-V2;

			% Update slopes
			z=zeros(BW,BR+1);
			alpha=zeros(BW,BR+1);
			for state=1:BW
				for level=1:BR+1
					if(state == sample(k))
						% Calculate alpha and z
						if(level == Rx(k))
							alpha(state,level)=1/max(max(NV(k:N:(BW-1)*N+k,:)));
							z(state,level)=(1-alpha(state,level))*v((state-1)*N+k,level)+alpha(state,level)*vhatlo;
						elseif(level == Rx(k)+1)
							alpha(state,level)=1/max(max(NV(k:N:(BW-1)*N+k,:)));
							z(state,level)=(1-alpha(state,level))*v((state-1)*N+k,level)+alpha(state,level)*vhatup;
						else
							alpha(state,level)=0;
							z(state,level)=v((state-1)*N+k,level);
						endif

						% Projection operation
						if(level < Rx(k) && z(state,level) <= z(sample(k),Rx(k)))
							v((sample(k)-1)*N+k,level)=z(sample(k),Rx(k));
						elseif(level > Rx(k)+1 && z(state,level) >= z(sample(k),Rx(k)+1))
							v((sample(k)-1)*N+k,level,k)=z(sample(k),Rx(k)+1);
						else
							v((sample(k)-1)*N+k,level,k)=z(state,level);
						endif
					endif
				endfor
			endfor
		endif
	endfor
endfor

plot(n,(Rx-1)/rho,n,W.g(1,:),n,W.r(1,:))

