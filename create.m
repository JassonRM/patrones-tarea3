## (C) 2020 Pablo Alvarado
## EL5852 Introducción al Reconocimiento de Patrones
## Tarea 3


## usage: [X,z] = create(numSamples[,noise])
##
## Create simulated depth data.  For each position in a row of X, the
## normalized z coordinate is returned.  0 means the deepest value and
## 1 the highest value.
##
## Optionally you can provide the level of desired data noise in the z
## direction, which by default is set to 1/256.
##
## The data created consists of 2D grid positions in the rows of X,
## and the corresponding z value at that position.  The order of the
## 2D positions in X is completely random (except for the reference data).
##
## The mandatory argument is a number, that indicates the absolute
## number of samples (if the provided number is greater than 1), or
## the percentage of total samples (if the provided number is less or
## equal than 1).
##
## Optionally you can provide how much noise you want on the depth
## measurement (noiseZ) and on the plane (noiseP)
##
function [X,z] = create(numSamples,noiseZ=1.0/256,noiseP=0.01)

  pkg load image;

  ## Load the 8-bit profile
  Orig = imread("heightmap.png");

  ## Cast it to double and pad it before filtering
  I = padarray(im2double(Orig),[2,2],"replicate");

  k5 = [1 4 6 4 1]; ## Binomial kernel
  k5 = k5 / sum(k5);

  ## Filter the profile to smooth out the 8-bit boundaries
  F=conv2(k5,k5,I,"valid");

  ## Get the complete data grid
  [xx,yy] = meshgrid(1:columns(F),1:rows(F));
  all = [xx(:) yy(:)];

  ## Does the user want the reference data only?
  if (numSamples==1 && noiseZ==0 && noiseP==0)
    ## Return the reference data
    X=all;
    z=F(:);
  else
    if numSamples<=1
      numSamples = round(numSamples*rows(Orig)*columns(Orig));
    endif

    ## Take some random data without replacement
    idx=randperm(rows(all),numSamples);

    ## Get the data and add the desired noise
    X = all(idx,:) + randn(rows(idx),2)*noiseP;
    z = F(idx) + randn(rows(idx),1)*noiseZ;
  endif
endfunction
