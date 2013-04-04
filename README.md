Notes 
===== 

- The size-mismatch issue with timesteps happens because I
choose the coarse time grid by taking the min and max of the times
read in from the data file. Then, when doing the integration, I do
choose the min and max of that NEW GRID, so if I do arange on those
values it won't include the last entry -- so I need to do
arange(t1,t2+dt,dt) in the integration.