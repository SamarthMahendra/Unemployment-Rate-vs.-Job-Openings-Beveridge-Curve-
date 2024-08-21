# Unemployment Rate vs. Job Openings (Beveridge Curve)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in the data
data = pd.read_csv('AmericanWorksData.csv')

# print all columns in the data
print(data.columns)

# Index(['Industry', 'Period', 'Employment (thousands)',
#        'Unemployment (thousands)',
#        'Unemployment compared to pre-pandemic levels (%)',
#        'Unemployment Rate (%)', 'Job Openings (thousands)',
#        'Job Openings compared to pre-pandemic levels (%)',
#        'Labor Shortage or Surplus (thousands)', 'Available Worker Ratio',
#        'Quits (thousands)', 'Quit Rate (%)', 'Hires (thousands)',
#        'Hire Rate (%)'],
#       dtype='object')
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Select the columns we need
data = data[['Period', 'Unemployment Rate (%)', 'Job Openings (thousands)']]
# Divide the job openings by 1000 to get the actual number
data['Job Openings (thousands)'] = data['Job Openings (thousands)'] / 1000
data['Period'] = pd.to_datetime(data['Period'])
# Group by period and get the mean of the unemployment rate and job openings
data = data.groupby('Period').mean()

# Sort by period (date)
data = data.sort_values(by='Period', ascending=False)

# Convert the data to tensor
x = torch.tensor(data['Job Openings (thousands)'].values).float()
y = torch.tensor(data['Unemployment Rate (%)'].values).float()

# Reshape the data to match the model input requirements
x = x.view(-1, 1)
y = y.view(-1, 1)

# Create a linear regression model using a quadratic function
class QuadraticRegression(nn.Module):
    def __init__(self):
        super(QuadraticRegression, self).__init__()
        # quadratic function: y = ax^2 + bx + c
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.cat((x, x ** 2), 1)
        return self.linear(x)


# Create a model
model = QuadraticRegression()

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train the model
epochs = 50000
for epoch in range(epochs):
    # Forward pass
    y_pred = model(x)

    # Compute the loss
    loss = criterion(y_pred, y)

    # Zero the gradients
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update the weights
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Generate a smooth range of x values for plotting the quadratic curve
x_range = torch.linspace(x.min().item(), x.max().item(), 100).view(-1, 1)
with torch.no_grad():
    y_range_pred = model(x_range)

# Convert predictions back to numpy for plotting
x_range = x_range.numpy()  # Convert x_range for consistency
y_range_pred = y_range_pred.numpy()  # Convert the predicted y values

# Plot the original data and the fitted quadratic curve
plt.figure(figsize=(10, 6))
plt.scatter(data['Job Openings (thousands)'], data['Unemployment Rate (%)'], color='blue', label='Data')
plt.plot(x_range, y_range_pred, color='red', label='Quadratic Fit')  # Plot the fitted curve using the smooth range
plt.title('Unemployment Rate vs Job Openings')
plt.xlabel('Job Openings (thousands)')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.show()



