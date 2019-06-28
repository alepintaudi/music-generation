midi_files_train, midi_files_val, midi_files_test  = list_midi_files(MIDI_SOURCE, DATA_LEN_TRAIN, DATA_LEN_VAL, DATA_LEN_TEST, randomSeq=True)

print(midi_files_train)
dataset_train=MidiDataset(qnsf=QNSF, seq_len=SEQ_LEN, cod_type=COD_TYPE, midi_files=midi_files_train)
print(len(dataset_train))
print(midi_files_val)
dataset_val=MidiDataset(qnsf=QNSF, seq_len=SEQ_LEN, cod_type=COD_TYPE, midi_files=midi_files_val)
print(len(dataset_val))
print(midi_files_test)
dataset_test=MidiDataset(qnsf=QNSF, seq_len=SEQ_LEN, cod_type=COD_TYPE, midi_files=midi_files_test)
print(len(dataset_test))

training_generator = data.DataLoader(dataset_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
validating_generator = data.DataLoader(dataset_val, batch_size=BATCH_SIZE_VAL, shuffle=True)
testing_generator = data.DataLoader(dataset_test, batch_size=BATCH_SIZE_TEST, shuffle=True)

for batch in training_generator:
    break
print(batch[0].shape)
for batch in validating_generator:
    break
print(batch[0].shape)
for batch in testing_generator:
    break
print(batch[0].shape)

