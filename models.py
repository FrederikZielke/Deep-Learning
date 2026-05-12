import torch

import torch.nn as nn

class CNN_Astronomy(nn.Module):
    def __init__(self, nLabels):
        super(CNN_Astronomy, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Dropout(0.2),
            nn.AvgPool1d(3),

            nn.Conv1d(8, 16, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            nn.AvgPool1d(3),

            nn.Conv1d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.AvgPool1d(3),

            nn.Conv1d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Conv1d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),

            nn.Conv1d(32, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.3),

            nn.Conv1d(16, 8, kernel_size=1),
            nn.Dropout(0.3),

            nn.Flatten(),
            
            nn.Linear(4800, 128),
            nn.ReLU(),
            nn.Linear(128, nLabels),

        )

    def forward(self, x):
        x = self.model(x)
        return x
    
class galah(nn.Module):
    def __init__(self, latent_dimension, input_length = 16384):
        super(galah, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=6),
            nn.ReLU(),
            nn.AvgPool1d(2),

            nn.Conv1d(16, 32, kernel_size=4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(2),

            nn.Conv1d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(2),


            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),


            nn.Conv1d(64, 16, kernel_size=1),
            nn.ReLU()
        )

        dummy_input = torch.zeros(1, 1, input_length)
        
        # 2. Pass it through the feature extractor
        with torch.no_grad():
            dummy_output = self.features(dummy_input)
            
        # 3. Calculate the total flattened size (Channels * Remaining Length)
        flattened_size = dummy_output.view(1, -1).size(1)

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dimension)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x