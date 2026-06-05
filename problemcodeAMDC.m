function [final]=problemcodeAMDC(N,d,K,a,b)
n=0:N;
A=zeros((N+1)^2);
c=zeros((N+1)^2);
theta=(2*n+1)*pi/(2*N+2);
r=cos(theta/2);
[R,THETA]=meshgrid(r,theta);
for l=1:(N+1)^2
    p=1;
    for k=0:N
        for m=0:N
A(l,p)=sum(hyperterm(-k:k,k,m,R(l),THETA(l)).*getgl(-k:k,a,b));
c(l,p)=doubleintC(k,m,R(l),THETA(l),d,K,a,b);
p=p+1;
        end

    end

end
C=A+c;
f=4*pi*ones((N+1)^2,1);
X=C\f;
[x1,w1]=lgwt(100,0,1);
[x2,w2]=lgwt(100,0,2*pi);
q=1;sum1=0;
for k=0:N
    for m=0:N
sum1=sum1+X(q)*(double(alform(k,m,x1)).*x1)*cos(m*x2).';
q=q+1;
    end
end
w=w1*w2';
final=-a*b*sum(sum(w.*sum1))/(pi*a*b);
