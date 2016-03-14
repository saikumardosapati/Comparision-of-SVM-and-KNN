images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
timages = loadMNISTImages('t10k-images.idx3-ubyte');
tlabels = loadMNISTLabels('t10k-labels.idx1-ubyte');

for i=1:784
    for j=1:60000
    img(j,i)=images(i,j);
    end
end

for i=1:784
    for j=1:10000
    timg(j,i)=timages(i,j);
    end
end


%positive and negative examples of class 0

idx0=[];
idx1=[];
idx2=[];
idx3=[];
idx4=[];
idx5=[];
idx6=[];
idx7=[];
idx8=[];
idx9=[];
for i =1:length(labels)
  
    if (labels(i)==0)
        idx0=[idx0 i];
    elseif (labels(i)==1)
         idx1=[idx1 i];
    elseif (labels(i)==2)
         idx2=[idx2 i];
    elseif (labels(i)==3)
        idx3=[idx3 i]; 
    elseif (labels(i)==4)
         idx4=[idx4 i];
    elseif (labels(i)==5)
         idx5=[idx5 i];
    elseif (labels(i)==6)
         idx6=[idx6 i];
    elseif (labels(i)==7)
         idx7=[idx7 i];
    elseif (labels(i)==8)
         idx8=[idx8 i];
    elseif (labels(i)==9)
         idx9=[idx9 i];
    end

end
%for training class 0  and the rest
%taking a sample of 100 positive and 100 negative in both training and test
%data build sample and calculate kernel values
totidx=[1:60000];
possample0=datasample(idx0,100,'replace',false);
negidx0=setdiff(totidx,idx0);
negsample0=datasample(negidx0,100,'replace',false);

possample1=datasample(idx1,100,'replace',false);
negidx1=setdiff(totidx,idx1);
negsample1=datasample(negidx1,100,'replace',false);

possample2=datasample(idx2,100,'replace',false);
negidx2=setdiff(totidx,idx2);
negsample2=datasample(negidx2,100,'replace',false);

possample3=datasample(idx3,100,'replace',false);
negidx3=setdiff(totidx,idx3);
negsample3=datasample(negidx3,100,'replace',false);

possample4=datasample(idx4,100,'replace',false);
negidx4=setdiff(totidx,idx4);
negsample4=datasample(negidx4,100,'replace',false);

possample5=datasample(idx5,100,'replace',false);
negidx5=setdiff(totidx,idx5);
negsample5=datasample(negidx5,100,'replace',false);

possample6=datasample(idx6,100,'replace',false);
negidx6=setdiff(totidx,idx6);
negsample6=datasample(negidx6,100,'replace',false);

possample7=datasample(idx7,100,'replace',false);
negidx7=setdiff(totidx,idx7);
negsample7=datasample(negidx7,100,'replace',false);

possample8=datasample(idx8,100,'replace',false);
negidx8=setdiff(totidx,idx8);
negsample8=datasample(negidx8,100,'replace',false);

possample9=datasample(idx9,100,'replace',false);
negidx9=setdiff(totidx,idx9);
negsample9=datasample(negidx9,100,'replace',false);


traindata0=[img(possample0,:);img(negsample0,:)];
%trainlabel0=[labels(possample0);labels(negsample0)];
%to see the distribution of negative examples
trainlabel0(1:100)=1;
trainlabel0(101:200)=-1;

traindata1=[img(possample1,:);img(negsample1,:)];
trainlabel1(1:100)=1;
trainlabel1(101:200)=-1;

traindata2=[img(possample2,:);img(negsample2,:)];
trainlabel2(1:100)=1;
trainlabel2(101:200)=-1;

traindata3=[img(possample3,:);img(negsample3,:)];
trainlabel3(1:100)=1;
trainlabel3(101:200)=-1;

traindata4=[img(possample4,:);img(negsample4,:)];
trainlabel4(1:100)=1;
trainlabel4(101:200)=-1;

traindata5=[img(possample5,:);img(negsample5,:)];
trainlabel5(1:100)=1;
trainlabel5(101:200)=-1;

traindata6=[img(possample6,:);img(negsample6,:)];
trainlabel6(1:100)=1;
trainlabel6(101:200)=-1;

traindata7=[img(possample7,:);img(negsample7,:)];
trainlabel7(1:100)=1;
trainlabel7(101:200)=-1;

traindata8=[img(possample8,:);img(negsample8,:)];
trainlabel8(1:100)=1;
trainlabel8(101:200)=-1;

traindata9=[img(possample9,:);img(negsample9,:)];
trainlabel9(1:100)=1;
trainlabel9(101:200)=-1;

%sampling test data

idx0=[];
idx1=[];
idx2=[];
idx3=[];
idx4=[];
idx5=[];
idx6=[];
idx7=[];
idx8=[];
idx9=[];
for i =1:length(tlabels)
  
    if (tlabels(i)==0)
        idx0=[idx0 i];
    elseif (tlabels(i)==1)
         idx1=[idx1 i];
    elseif (tlabels(i)==2)
         idx2=[idx2 i];
    elseif (tlabels(i)==3)
        idx3=[idx3 i]; 
    elseif (tlabels(i)==4)
         idx4=[idx4 i];
    elseif (tlabels(i)==5)
         idx5=[idx5 i];
    elseif (tlabels(i)==6)
         idx6=[idx6 i];
    elseif (tlabels(i)==7)
         idx7=[idx7 i];
    elseif (tlabels(i)==8)
         idx8=[idx8 i];
    elseif (tlabels(i)==9)
         idx9=[idx9 i];
    end

end
totidx=[1:10000];
possample0=datasample(idx0,100,'replace',false);
negidx0=setdiff(totidx,idx0);
negsample0=datasample(negidx0,100,'replace',false);

possample1=datasample(idx1,100,'replace',false);
negidx1=setdiff(totidx,idx1);
negsample1=datasample(negidx1,100,'replace',false);

possample2=datasample(idx2,100,'replace',false);
negidx2=setdiff(totidx,idx2);
negsample2=datasample(negidx2,100,'replace',false);

possample3=datasample(idx3,100,'replace',false);
negidx3=setdiff(totidx,idx3);
negsample3=datasample(negidx3,100,'replace',false);

possample4=datasample(idx4,100,'replace',false);
negidx4=setdiff(totidx,idx4);
negsample4=datasample(negidx4,100,'replace',false);

possample5=datasample(idx5,100,'replace',false);
negidx5=setdiff(totidx,idx5);
negsample5=datasample(negidx5,100,'replace',false);

possample6=datasample(idx6,100,'replace',false);
negidx6=setdiff(totidx,idx6);
negsample6=datasample(negidx6,100,'replace',false);

possample7=datasample(idx7,100,'replace',false);
negidx7=setdiff(totidx,idx7);
negsample7=datasample(negidx7,100,'replace',false);

possample8=datasample(idx8,100,'replace',false);
negidx8=setdiff(totidx,idx8);
negsample8=datasample(negidx8,100,'replace',false);

possample9=datasample(idx9,100,'replace',false);
negidx9=setdiff(totidx,idx9);
negsample9=datasample(negidx9,100,'replace',false);


testdata0=[timg(possample0,:);timg(negsample0,:)];
testlabel0(1:100)=1;
testlabel0(101:200)=-1;

testdata1=[timg(possample1,:);timg(negsample1,:)];
testlabel1(1:100)=1;
testlabel1(101:200)=-1;

testdata2=[timg(possample2,:);timg(negsample2,:)];
testlabel2(1:100)=1;
testlabel2(101:200)=-1;

testdata3=[timg(possample3,:);timg(negsample3,:)];
testlabel3(1:100)=1;
testlabel3(101:200)=-1;

testdata4=[timg(possample4,:);timg(negsample4,:)];
testlabel4(1:100)=1;
testlabel4(101:200)=-1;

testdata5=[timg(possample5,:);timg(negsample5,:)];
testlabel5(1:100)=1;
testlabel5(101:200)=-1;

testdata6=[timg(possample6,:);timg(negsample6,:)];
testlabel6(1:100)=1;
testlabel6(101:200)=-1;

testdata7=[timg(possample7,:);timg(negsample7,:)];
testlabel7(1:100)=1;
testlabel7(101:200)=-1;

testdata8=[timg(possample8,:);timg(negsample8,:)];
testlabel8(1:100)=1;
testlabel8(101:200)=-1;

testdata9=[timg(possample9,:);timg(negsample9,:)];
testlabel9(1:100)=1;
testlabel9(101:200)=-1;


%calculating kernel for each class label and rest

%0 and rest
var=0.8;
for i=1:200
    for j=1:200
        diff=traindata0(i,:)-testdata0(j,:);
        squared_diff=dot(diff,diff);
        kernel0(i,j)=exp(squared_diff*(-1)/(2*var*var));
    end
end

%1 and rest
var=0.8;
for i=1:200
    for j=1:200
        diff=traindata1(i,:)-testdata1(j,:);
        squared_diff=dot(diff,diff);
        kernel1(i,j)=exp(squared_diff*(-1)/(2*var*var));
    end
end

%2 and rest
var=0.8;
for i=1:200
    for j=1:200
        diff=traindata2(i,:)-testdata2(j,:);
        squared_diff=dot(diff,diff);
        kernel2(i,j)=exp(squared_diff*(-1)/(2*var*var));
    end
end

%3 and rest
var=0.8;
for i=1:200
    for j=1:200
        diff=traindata3(i,:)-testdata3(j,:);
        squared_diff=dot(diff,diff);
        kernel3(i,j)=exp(squared_diff*(-1)/(2*var*var));
    end
end

%4 and rest
var=0.8;
for i=1:200
    for j=1:200
        diff=traindata4(i,:)-testdata4(j,:);
        squared_diff=dot(diff,diff);
        kernel4(i,j)=exp(squared_diff*(-1)/(2*var*var));
    end
end

%5 and rest
var=0.8;
for i=1:200
    for j=1:200
        diff=traindata5(i,:)-testdata5(j,:);
        squared_diff=dot(diff,diff);
        kernel5(i,j)=exp(squared_diff*(-1)/(2*var*var));
    end
end

%6 and rest
var=0.8;
for i=1:200
    for j=1:200
        diff=traindata6(i,:)-testdata6(j,:);
        squared_diff=dot(diff,diff);
        kernel6(i,j)=exp(squared_diff*(-1)/(2*var*var));
    end
end

%7 and rest
var=0.8;
for i=1:200
    for j=1:200
        diff=traindata7(i,:)-testdata7(j,:);
        squared_diff=dot(diff,diff);
        kernel7(i,j)=exp(squared_diff*(-1)/(2*var*var));
    end
end

%8 and rest
var=0.8;
for i=1:200
    for j=1:200
        diff=traindata8(i,:)-testdata8(j,:);
        squared_diff=dot(diff,diff);
        kernel8(i,j)=exp(squared_diff*(-1)/(2*var*var));
    end
end

%9 and rest
var=0.8;
for i=1:200
    for j=1:200
        diff=traindata9(i,:)-testdata9(j,:);
        squared_diff=dot(diff,diff);
        kernel9(i,j)=exp(squared_diff*(-1)/(2*var*var));
    end
end
eps=0.00005;
[weight0, bias0, tp0, fp0,alpha0,tn0,fn0]=mysmo(testdata0,testlabel0,traindata0,trainlabel0,kernel0,eps);

[weight1, bias1, tp1, fp1,alpha1,tn1,fn1]=mysmo(testdata1,testlabel1,traindata1,trainlabel1,kernel1,eps);

[weight2, bias2, tp2, fp2,alpha2,tn2,fn2]=mysmo(testdata2,testlabel2,traindata2,trainlabel2,kernel2,eps);

[weight3, bias3, tp3, fp3,alpha3,tn3,fn3]=mysmo(testdata3,testlabel3,traindata3,trainlabel3,kernel3,eps);

[weight4, bias4, tp4, fp4,alpha4,tn4,fn4]=mysmo(testdata4,testlabel4,traindata4,trainlabel4,kernel4,eps);

[weight5, bias5, tp5, fp5,alpha5,tn5,fn5]=mysmo(testdata5,testlabel5,traindata5,trainlabel5,kernel5,eps);

[weight6, bias6, tp6, fp6,alpha6,tn6,fn6]=mysmo(testdata6,testlabel6,traindata6,trainlabel6,kernel6,eps);

[weight7, bias7, tp7, fp7,alpha7,tn7,fn7]=mysmo(testdata7,testlabel7,traindata7,trainlabel7,kernel7,eps);


[weight8, bias8, tp8, fp8,alpha8,tn8,fn8]=mysmo(testdata8,testlabel8,traindata8,trainlabel8,kernel8,eps);


[weight9, bias9, tp9, fp9,alpha9,tn9,fn9]=mysmo(testdata9,testlabel9,traindata9,trainlabel9,kernel9,eps);

accuracy=(tp0+tp1+tp2+tp3+tp4+tp5+tp6+tp7+tp8+tp9+tn0+tn1+tn2+tn3+tn4+tn5+tn6+tn7+tn8+tn9)/2000;
%accuracy for var=0.8 and loop 400 0.7520 i.e 75.20%
