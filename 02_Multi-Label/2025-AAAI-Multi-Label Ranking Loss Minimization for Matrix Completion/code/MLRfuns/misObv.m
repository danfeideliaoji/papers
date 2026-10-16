function [Z1, posSelect, Z2, posDump, posD] = misObv(Data, sr)
% sr: Sampling ratio
% posD: unmissing indices for Data, sum(posD)=num(Data)
% posSelect: obseved indices for Z, sum(posZ1)=num(Z)=sr*num(Data)
% posDump: unobseved indices for Z, sum(posZ2)+sum(posB)=num(Data), i.e., sum(posZ2):sum(posB)=(1-sr):sr

orgData = Data;
posD = double(Data ~= 0);
num_nonzero = sum(sum(Data ~= 0));
num_remove = floor(num_nonzero * (1 - sr));
pos_nonzero = find(Data ~= 0);
remove = datasample(pos_nonzero, num_remove, 'Replace', false);
Data(remove) = 0;
Z1 = Data;
posSelect = double(Z1 ~= 0);
posDump = posD - posSelect;
Z2 = orgData - Z1;
end