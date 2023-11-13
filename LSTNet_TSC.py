import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from LSTNet import LSTNetForClassification
import time
from utils import *
import Optim

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def get_shapes(np_arr_list):
    """Returns text, each line is shape and dtype for numpy array in list
       example: print(get_shapes([X_train, X_test, y_train, y_test]))"""
    shapes = ""
    for i in np_arr_list:
        my_name = namestr(i,globals())
        shapes += (my_name[0] + " shape is " + str(i.shape) \
            + " data type is " + str(i.dtype) + "\n")
    return shapes

    dataset = "UniMiB SHAR"
    datasetfn = 'mobiact_adl_load_dataset.py'
    full_filename = 'unimib_shar_adl_load_dataset.py'

    #x_train, y_train, x_test, y_test = unimib_load_dataset(incl_val_group = False)
    x_train, y_train, x_validate, y_validate, x_test, y_test = unimib_load_dataset(incl_val_group = True)
    t_names = ['StandingUpFS','StandingUpFL','Walking','Running','GoingUpS','Jumping','GoingDownS','LyingDownFS','SittingDown']

  
    print(get_shapes([x_train, y_train, x_validate, y_validate, x_test, y_test]))


def evaluate_classification(data, X, Y, model, batch_size):
    model.eval()
    total_correct = 0
    total_samples = 0

    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X)
        _, predicted_labels = torch.max(output, 1)
        total_correct += (predicted_labels == Y).sum().item()
        total_samples += Y.size(0)

    accuracy = total_correct / total_samples
    return accuracy

def train_classification(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    total_samples = 0

    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        grad_norm = optim.step()
        total_loss += loss.item()
        total_samples += Y.size(0)

    return total_loss / total_samples

batch_size = 32 
input_size = 6  
hidden_size = 50
output_size = 9  
epochs = 100
lr = 0.001

# Create model, optimizer, and criterion
model = LSTNetForClassification(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    for i in range(len(x_train)):
        x, y = x_train[i:i+1], y_train[i:i+1]
        loss = train(model, optimizer, criterion, x, y)

    if epoch % 10 == 0:
        accuracy = evaluate(model, x_test, y_test)
        print(f'Epoch {epoch}: Test Accuracy: {accuracy:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'simple_model.pth')

print(dataset, "1D CNN")
print("Final Validation Accuracy: %0.3f" % history.history['val_accuracy'][-1])
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.ylim([0,1.2]) #set limit - 1.2 has been a good value experimentally
plt.legend()
plt.show()

predictions = model.predict(x_test, verbose=0,batch_size=32)

#must use values not one-hot encoding, use argmax to convert
y_pred = np.argmax(predictions, axis=-1) # axis=-1 means last axis
y_test_act = np.argmax(y_test, axis=-1) # undo one-hot encoding

# Print print prediction accuracy
print('Prediction accuracy: {0:.3f}'.format(accuracy_score(y_test_act, y_pred)))

# Print a report of classification performance metrics
print(classification_report(y_test_act, y_pred, target_names=t_names))

# Plot a confusion matrix
cm = confusion_matrix(y_test_act, y_pred)
cm_df = pd.DataFrame(cm,
                     index = t_names, 
                     columns = t_names)
fig = plt.figure(figsize=(6.5,5))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='cubehelix_r')
plt.title('1D CNN using '+dataset+'\nAccuracy:{0:.3f}'.format(accuracy_score(y_test_act, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout() # keeps labels from being cutoff when saving as pdf
plt.show()
