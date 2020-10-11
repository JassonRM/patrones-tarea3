## Example of creating and showing the sensor raw data

## Use for the experiments just 0,5% of the total available data.
[X,z] = create(0.005);

plot3(X(:,1),X(:,2),z(:),".");
xlabel("x")
ylabel("y")
zlabel("z")

