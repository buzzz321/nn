# Prevent Octave from thinking that this
# is a function file:
# heavly inspired of tsodongs nn series
1;

# activation function
function g = sigmoid(z)
  g = 1 ./ (1 + exp(-z));
end

% makes my nn struct
function Xor  = makestruct()
  mystruct.a0=zeros(1,2);

  mystruct.w1=rand(2,2);
  mystruct.b1=rand(1,2);
  mystruct.a1=rand(1,2);
  
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
function Xor = forward(Xor)
  
  Xor.a1=Xor.a0*Xor.w1;
  Xor.a1=Xor.a1+Xor.b1; % bias
  Xor.a1=sigmoid(Xor.a1);% atcivate with sigmoid to make a1 be values betwee 0 and 1

  Xor.a2=Xor.a1*Xor.w2;
  Xor.a2=Xor.a2+Xor.b2; %bias 
  Xor.a2=sigmoid(Xor.a2);% atcivate with sigmoid to make a2 be values betwee 0 and 1

end

function model = finit_diff(model, gradient, trainInput, trainOutput, eps)
  [model c]= cost(model, trainInput, trainOutput);
  
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

Xor=forward(Xor);

disp(Xor.a2);
