N=1000;

v=normrnd(0,2,N);

b=[1];
a=[1, -0.6];

y=filter(b,a,v)
r=autocorr(y);
n=linspace(0,N,N);

w=linspace(0,2*pi,N);
Y=fft(y);
H=freqz(b,a,w);
PSD=PSD_gen(H*Y);

figure
plot(n,y);
figure
plot(n,r);
figure
plot(w,PSD)



function r=autocorr(y)
    N=length(y);
    r=zeros(1,N);
    for k=0:(N-1)
        for i=1:(N-k)
            r(1,k+1)=r(1,k+1) + y(1,i)*y(1,i+k);
        end
    end
    r=r/N;
end

function PSD=PSD_gen(y)
    N=length(y);
    PSD=abs(y).^2;
    PSD=PSD/N;
end




