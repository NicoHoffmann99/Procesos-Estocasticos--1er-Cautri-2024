N=49;

F=[0 0.45 0.5 1];
A=[1 1 0 0];
V=[1/0.04 1/0.04];

b=firpm(N,F,A,V)

[H, w] = freqz(b,1);
figure
plot(w/pi,abs(H));

figure
zplane(b)


N_2=45;

F_2=[0 0.25 0.3 0.65 0.7 1];
A_2=[0 0 1 1 0 0];
V_2=[1/0.06 1/0.1 1/0.03];

b_2=firpm(N_2,F_2,A_2,V_2);

[H_2, w_2] = freqz(b_2,1);
figure
plot(w_2/pi,abs(H_2));

figure
zplane(b_2)


