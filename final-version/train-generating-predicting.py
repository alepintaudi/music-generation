model = Seq2Seq(88*COD_TYPE)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for i,batch in enumerate(training_generator):
    x = batch[0]
    y = batch[1]
    x= x.to(device)
    y= y.to(device)

    y_pred = model.train()(x,y,teacher_forcing_ratio=0.5*(1-epoch/EPOCHS))
    break
    
(y_pred>0).float().mean()

torch.mul((y_pred>0)[0][-1].float(),y[0][-1]).sum()/y[0][-1].sum()

y[0][-1].sum()

%%time
# init model

# trainin validation

loss_train=[]
acc_train=[]
loss_val=[]
acc_val=[]

for epoch in range(EPOCHS):
  for i,batch in enumerate(training_generator):
        # Forward pass: Compute predicted y by passing x to the model
        x = batch[0]
        y = batch[1]
        x= x.to(device)
        y= y.to(device)

        y_pred = model.train()(x,y,teacher_forcing_ratio=0.5*(1-epoch/EPOCHS))

        # Compute and print loss
        loss = model.focal_loss(y_pred, y)
        acc  = model.accuracy(y_pred, y)


        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
  loss_train.append(loss.item())
  acc_train.append(acc.item())
  
  # break
  
  if epoch%10==0:
    for i, batch in enumerate(validating_generator):
          loss_list = []
          acc_list = []
          x = batch[0]
          y = batch[1]
          x= x.to(device)
          y= y.to(device)
          y_pred = model.eval()(x,y,0)   #teacher forcing set to 0 during validation

          # Compute and print loss
          loss = model.focal_loss(y_pred, y)
          acc  = model.accuracy(y_pred, y)
          loss_list.append(loss.item())
          acc_list.append(acc.item())
  else:
    loss_list.append(loss_list[-1])
    acc_list.append(acc_list[-1])
    
  loss_val.append(np.asarray(loss_list).mean())
  acc_val.append(np.asarray(acc_list).mean())

  print(epoch, loss_train[epoch], loss_val[epoch], acc_train[epoch], acc_val[epoch])

seq2seqMod = model 

seq2seqMod = model

torch.Size([50, 64, 176])
torch.Size([50, 64, 512])
torch.Size([50, 64, 512])
torch.Size([50, 64, 1, 512])
torch.Size([50, 64, 1, 176])
torch.Size([50, 64, 176])

seq_pred=seq2seqMod.predict()
