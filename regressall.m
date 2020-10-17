## (C) 2020 Pablo Alvarado
## EL5852 Introducción al Reconocimiento de Patrones
## Tarea 3

## Template file for the whole solution

1;

## Linear regression
##
## p: matrix of size m x 2, with m 2D positions on which
##    the z value needs to be regressed
## X: support data (or training data) with all known 2D positions
## z: support data with the corresponding z values for each position
##
## The number of rows of X must be equal to the length of y
##
## The function must generate the z position for all
function rz=lr(p,X,z)
  X = [ones(length(X), 1), X];
  theta=pinv(X)*z(:);
  p = [ones(length(p), 1), p];
  rz=p*theta;
endfunction

## Locally weighted regression
##
## p: matrix of size m x 2, with m 2D positions on which
##    the z value needs to be regressed
## X: support data (or training data) with all known 2D positions
## z: support data with the corresponding z values for each position
##
## The number of rows of X must be equal to the length of y
##
## The function must generate the z position for all
function rz=lwr(p,X,z,tau)
  rz = [];
  sx = rows(X);
  ntau = 2*tau**2;
  X = [ones(length(X), 1), X];
  p = [ones(length(p), 1), p];
  for i = 1:rows(p)
    W = ones(sx,1)*p(i,:) - X;
    W = vecnorm(W,2,2).^2 / -ntau;
    W = exp(W);
    theta = pinv(X'*(W.*X))*X'*(W.*z(:));
    rz = [rz; p(i,:)*theta];
  endfor

endfunction

function error=loss(reference, regressed)

  ## Log-Cosh
  temp = log(cosh(regressed - reference));
  error = sum(temp);

  ## MSE
  % R = (regressed - reference);
  % n = size(R)(1);
  % error = sum(R.*R) / n;
endfunction

## Use for the experiments just 0,5% of the total available data.
[X,z] = create(0.005);

## And the reference data to be used for comparison
[RX,rz] = create(1,0,0);

## Extract from RX the used coordinate range:
minx=min(RX(:,1))
maxx=max(RX(:,1))

miny=min(RX(:,2))
maxy=max(RX(:,2))

partition=75;
[xx,yy]=meshgrid(round(linspace(minx,maxx,partition)),
                 round(linspace(miny,maxy,partition)));

## The grid
NX = [xx(:) yy(:)];
indices = sub2ind([maxy, maxx], NX(:,2), NX(:,1));
rnz = rz(indices);

printf("Linear regression started...");
fflush(stdout);
tic();

lrz = lr(NX, X, z);

printf("done.\n");
toc()
fflush(stdout);

## Locally weighed regression on the data
## This will take a LONG time once finished
printf("Weighed regression started...");
fflush(stdout);
tic();

#soft curve
tau=32;
nz = lwr(NX,X,z,tau);

printf("done.\n");
toc()
fflush(stdout);

## Calculate the error for different values of tau
lower_error = 1e10;
best_z = [];
best_tau = 0;
errors = [];
taus = [];
for tau=1:2:30
  nz1 = lwr(NX, X, z, tau);
  err = loss(rnz, nz1);
  taus = [taus tau];
  errors = [errors err];
  if err < lower_error
    lower_error = err;
    best_z = nz1;
    best_tau = tau;
  endif
endfor

printf("El mejor tau es %d\n", best_tau);

## Plot all the data and results


figure(1,"name","Sensed data");
plot3(X(:,1),X(:,2),z',".");
xlabel("x")
ylabel("y")
zlabel("z")

figure(2,"name","Regressed data (LWR)");
hold off;
#plot3(X(:,1),X(:,2),y',"b.");
#hold on;
plot3(NX(:,1),NX(:,2),nz,"r.");
#surf(xx,yy,reshape(ny,size(xx)));
xlabel("x")
ylabel("y")
zlabel("z")
title("Soft linear weighted regression")

figure(3,"name","Regressed data (LR)");
hold off;
#plot3(X(:,1),X(:,2),y',"b.");
#hold on;
plot3(NX(:,1),NX(:,2),lrz,"r.");
#surf(xx,yy,reshape(ny,size(xx)));
xlabel("x")
ylabel("y")
zlabel("z")
title("Linear regression")

figure(4,"name","Best Tau (LWR)");
hold off;
#plot3(X(:,1),X(:,2),y',"b.");
#hold on;
plot3(NX(:,1),NX(:,2),best_z,"r.");
#surf(xx,yy,reshape(ny,size(xx)));
xlabel("x")
ylabel("y")
zlabel("z")
title(strcat("Best tau value = ", num2str(best_tau)))

figure(5,"name","Error");
plot(taus, errors, "linewidth", 3);
xlabel("tau")
ylabel("error")
