M=1000;

V=normrnd(0,4,1,M);

r=zeros(1,M);

for k=0:(M-1)
    for i=1:(M-k)
        r(1,k+1)=r(1,k+1) + V(1,i)*V(1,i+k);
    end
end
r=r/M;
%r_i=xcorr(V,'biased');
x_1=linspace(1,M,M);
%x_2=linspace(1,length(r_i),length(r_i));
figure
plot(x_1,r);
%plot(x_2,r_i);

x_3=linspace(0,2,M);
PSD=abs(fft(V)).^2;
PSD=PSD/M;
figure
plot(x_3,PSD);




    







