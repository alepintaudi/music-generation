epochs = range(EPOCHS)

loss_obj=[0]*EPOCHS

plt.figure(figsize=(10,6))
plt.plot(epochs, loss_train, '-o', label='Training loss')
plt.plot(epochs, loss_val, '-o', label='Validation loss')
plt.plot(epochs, loss_obj, '--')

plt.legend()
plt.title('Learning curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

epochs = range(EPOCHS)

acc_obj=[1]*EPOCHS

plt.figure(figsize=(10,6))
plt.plot(epochs, acc_train, '-o', label='Training accuracy')
plt.plot(epochs, acc_val, '-o', label='Validation accuracy')
plt.plot(epochs, acc_obj, '--')

plt.legend()
plt.title('Learning curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
