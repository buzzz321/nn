% Prevent Octave from thinking that this
% is a funktion file
% heavly inspired of tsodings nn series
1;

% activation function
function g = sigmoid(z)
  g = 1 ./ (1 + exp(-z));
end

% makes my nn struct
function Xor  = makestruct()
  mystruct.a0=zeros(1,2);

  mystruct.w1=rand(2,2); % weight
  mystruct.b1=rand(1,2); % bias
  mystruct.a1=rand(1,2); % activation

  mystruct.w2=rand(2,1);
  mystruct.b2=rand(1,1);
  mystruct.a2=zeros(1,1);
  Xor = mystruct;
end

% cost function
function [retVal Xor] = cost(mat, trainInput, trainOutput)
  n = size(trainInput,1);

  c = 0.0; % our cost
  for i = 1:n
    x = trainInput(i,:);
    y = trainOutput(i,:);

    mat.a0 = x;
    mat = forward(mat);

    diff = mat.a2(1,:) - y(1,:);
    c = c + diff*diff.'; % matlab way square a matrix
  end

  Xor = mat;
  retVal = c/n;
end

% forward function
function retVal = forward(Xor)

  Xor.a1=Xor.a0*Xor.w1;
  Xor.a1=Xor.a1+Xor.b1; % bias
  Xor.a1=sigmoid(Xor.a1);% atcivate with sigmoid to make a1 be values betwee 0 and 1

  Xor.a2=Xor.a1*Xor.w2;
  Xor.a2=Xor.a2+Xor.b2; %bias
  Xor.a2=sigmoid(Xor.a2);% atcivate with sigmoid to make a2 be values betwee 0 and 1

  retVal =  Xor;
  %disp(retVal);
end

function retVal = finit_diff(model, trainInput, trainOutput, eps)
  [c model] = cost(model, trainInput, trainOutput);
  
  [rows cols] = size(model.w1);
  for i = 1:rows
    for j = 1:cols
      saved = model.w1(i,j);
      model.w1(i,j) += eps;
      model.g1(i,j) = (cost(model, trainInput,trainOutput) - c) /eps;
      model.w1(i,j) = saved;
    end
  end

  [rows cols] = size(model.b1);
  for i = 1:rows
    for j = 1:cols
      saved = model.b1(i,j);
      model.b1(i,j) += eps;
      model.g1(i,j) = (cost(model, trainInput,trainOutput) - c) /eps;
      model.b1(i,j) = saved;
    end
  end

  [rows cols] = size(model.w2);
  for i = 1:rows
    for j = 1:cols
      saved = model.w2(i,j);
      model.w2(i,j) += eps;
      model.g2(i,j) = (cost(model, trainInput,trainOutput) - c) /eps;
      model.w2(i,j) = saved;
    end
  end

  [rows cols] = size(model.b2);
  for i = 1:rows
    for j = 1:cols
      saved = model.b2(i,j);
      model.b2(i,j) += eps;
      model.g2(i,j) = (cost(model, trainInput,trainOutput) - c) /eps;
      model.b2(i,j) = saved;
    end
  end

  retVAl = model;
end

Xor=makestruct();

trainData = [ 0, 0, 0;
              0, 1, 1;
              1, 0, 1;
              1, 1, 0 ];

trainInput = trainData(1:4,1:2);
trainOutput = trainData(1:4,3);
printf("--------------------------\n");
[c ~] = cost(Xor, trainInput, trainOutput);
disp(c);
printf("--------------------------\n");
Xor.a0(1,1)=0;
Xor.a0(1,2)=1;

Xor = forward(Xor);

disp(Xor.a2);

eps = 1.0;

finit_diff(Xor, trainInput, trainOutput, eps)
