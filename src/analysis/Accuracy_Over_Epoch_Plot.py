import pandas as pd
import matplotlib.pyplot as plt

#headers_timm = ['epoch',	'train_loss',	'eval_loss',	'eval_top1',	'eval_top5',	'lr']
headers = ['Training loss','Training acc','validation loss', 'validation acc']
df = pd.read_csv('summary_50.csv',names=headers, skiprows=[0]) #
#df.iloc[0: 50]
#print(df)
#print(df.iloc[0: 50])

#df['Date'] = df['Date'].map(lambda x: datetime.strptime(str(x), '%Y/%m/%d %H:%M:%S.%f'))
train_losses = df['Training loss']
test_losses = df['Training acc']
#print (train_losses)
# Prepare data
x = [i for i in range(1, 50)]

# Create a plot
plt.plot(train_losses, color='green', alpha=0.8, label='Training Accuracy')
plt.plot(test_losses, color='magenta', alpha=0.8, label='Validation Loss')

# Add plot title and axis labels
plt.title("ShuffleNetV2_x0_5 PyTorch (SGD Optimizer)", fontsize=14)
plt.xlabel("Epochs")
plt.ylabel("Accuracy over epochs")

#Add the legent on the uper left corner
plt.legend(loc='lower right')

# Save the plot to a file
plt.savefig("ShuffleNetV2_x0_5_training_plot.png")

# Show plot
from PIL import Image

# Show plot
plt.figure()
plt.show()

# Open the PNG file
im = Image.open("ShuffleNetV2_x0_5_training_plot.png")

# Display the image
im.show()