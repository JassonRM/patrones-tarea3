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
## y: support data with the corresponding z values for each position
##
## The number of rows of X must be equal to the length of y
##
## The function must generate the z position for all
function rz=lr(p,X,z)
  theta=pinv(X)*z(:);
  rz=p*theta;
endfunction

## Locally weighted regression
##
## p: matrix of size m x 2, with m 2D positions on which
##    the z value needs to be regressed
## X: support data (or training data) with all known 2D positions
## y: support data with the corresponding z values for each position
##
## The number of rows of X must be equal to the length of y
##
## The function must generate the z position for all
function rz=lwr(p,X,z,tau)
  ## This code is for simple linear regression

  ## CHANGE THE FOLLOWING CODE
  ## You have to replace it for proper LWR code
  theta=pinv(X)*z(:);
  rz=p*theta;
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

printf("Regression started...");
fflush(stdout);
tic();

lrz = lr(NX, X, z);

printf("done.\n");
toc()
fflush(stdout);

figure(1,"name","Sensed data");
plot3(X(:,1),X(:,2),z',".");
xlabel("x")
ylabel("y")
zlabel("z")


figure(2,"name","Regressed data");
hold off;
#plot3(X(:,1),X(:,2),y',"b.");
#hold on;
plot3(NX(:,1),NX(:,2),lrz,"r.");
#surf(xx,yy,reshape(ny,size(xx)));
xlabel("x")
ylabel("y")
zlabel("z")
title("Linear regression")

## Locally weighed regression on the data
## This will take a LONG time once finished
tau=5;
nz = lwr(NX,X,z,tau);

printf("done.\n");
toc()
fflush(stdout);

figure(1,"name","Sensed data");
plot3(X(:,1),X(:,2),z',".");
xlabel("x")
ylabel("y")
zlabel("z")


figure(2,"name","Regressed data");
hold off;
#plot3(X(:,1),X(:,2),y',"b.");
#hold on;
plot3(NX(:,1),NX(:,2),nz,"r.");
#surf(xx,yy,reshape(ny,size(xx)));
xlabel("x")
ylabel("y")
zlabel("z")
title("Once fixed, this shouldn't be a plane") ## REMOVE ME WHEN DONE
