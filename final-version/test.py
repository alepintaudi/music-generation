# test

loss_test=[]
acc_test=[]

for i,batch in enumerate(testing_generator):
  x = batch[0]
  y = batch[1]
  x= x.to(device)
  y= y.to(device)
  y_pred = seq2seqMod.eval()(x,y,0)    #teacher forcing set to 0 during testing
  
  # Compute and print loss
  loss = seq2seqMod.loss(y_pred, y)
  acc  = seq2seqMod.accuracy(y_pred, y)
  
  loss_test.append(loss.item())
  acc_test.append(acc.item())

print(np.asarray(loss_test).mean(), np.asarray(acc_test).mean())
print(np.asarray(loss_test).std(), np.asarray(acc_test).std())
