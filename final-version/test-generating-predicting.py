
def generateSeq2seq(x, y, model ,pred_seq=1):
  z= x.to(device)
  w= y.to(device)
  x1shape = x.shape[1]
  pred_shape=list(x.shape[1:])
  pred_shape[0] += pred_seq*pred_shape[0]
  pred_test=np.zeros(pred_shape)
  pred_test[0:x1shape,:] = x[0,:,:].numpy()                # we take the firdt samples from x 
  for i in range(pred_seq):                                # predicting 
    y_pred = model.eval()(z,w,0)     #teacher forcing set to 0 during validation
    y_pred_pos=(y_pred[0,:,:].cpu().detach().numpy()>0)
    for ii in range(len(y_pred_pos)):
      pred_test[x1shape + ii,y_pred_pos[ii]] = 1.0
  return pred_test
  
  # generated length
PRED_SEQ = 1    #   !!!!  do not change it
dataset_test2=MidiDataset(qnsf=QNSF, seq_len=SEQ_LEN, cod_type=COD_TYPE, midi_files=['Classical-Piano-Composer/midi_songs/Gold_Silver_Rival_Battle.mid'])


# we recover one song for the first sequence
pred_generator = data.DataLoader(dataset_train, batch_size=1, shuffle=False)
# pred

# transform to input tensor last batch
for i,batch in enumerate(pred_generator):
  xinit = batch[0][0:1,:,:]
  yinit = batch[1][0:1,:,:]
  break

print(xinit.shape)
print(yinit.shape)

y_pred = seq2seqMod.eval()(xinit.to(device),yinit.to(device),0)


'''

pred_test = generateSeq2seq(xinit,yinit,seq2seqMod,pred_seq=PRED_SEQ)
print(pred_test.shape)
if COD_TYPE==2:
  pred_test_clean = cleanSeq(pred_test, cod_type=COD_TYPE)
else:
  pred_test_clean = pred_test
print(pred_test_clean.shape)
create_midi(pred_test_clean, qnsf = QNSF, cod_type=COD_TYPE, midiOutputFile='test_seq2seqv2_006.mid')

'''

pred_test = generateSeq2seq(xinit,yinit,seq2seqMod,pred_seq=PRED_SEQ)

create_midi(pred_test, qnsf = QNSF, cod_type=COD_TYPE, midiOutputFile='test_seq2seqv2_004.mid')
