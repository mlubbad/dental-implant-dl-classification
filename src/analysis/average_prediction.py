import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load your pre-trained models
model1 = regnet_y_32gf()
model1.load_state_dict(torch.load("path_to_model1_weights"))
model2 = regnet_y_16gf()
model2.load_state_dict(torch.load("path_to_model2_weights"))

# Load your validation data and labels
validation_data = ...
validation_label = ...

# create data loader
validation_loader = DataLoader(validation_data, batch_size=batch_size)

# Set models in evaluation mode
model1.eval()
model2.eval()

# initialize variables
predictions1 = []
predictions2 = []

# get predictions from models
with torch.no_grad():
    for data in validation_loader:
        output1 = model1(data)
        predictions1.append(output1)
        output2 = model2(data)
        predictions2.append(output2)

# concatenate predictions to create new dataset
new_dataset = torch.cat((torch.cat(predictions1), torch.cat(predictions2)), dim=1)

# Average the predictions
# Step 3: Average the predictions
average_predictions = new_dataset.mean(dim=1)

# Step 4: Use the averaged predictions as the final prediction
final_prediction = torch.argmax(average_predictions, dim=1)

#Convert the final prediction and validation_label to numpy array for evaluation metrics
final_prediction = final_prediction.numpy()
validation_label = validation_label.numpy()


# Step 5: calculate the accuracy, precision, recall and F1-score
acc = accuracy_score(validation_label, final_prediction)
prec = precision_score(validation_label, final_prediction, average='weighted')
rec = recall_score(validation_label, final_prediction, average='weighted')
f1 = f1_score(validation_label, final_prediction, average='weighted')
print("Accuracy: {:.4f}".format(acc))
print("Precision: {:.4f}".format(prec))
print("Recall: {:.4f}".format(rec))
print("F1-score: {:.4f}".format(f1))
