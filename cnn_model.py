import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import psutil

class CrimeDataset(Dataset):
    def __init__(self, X_text, X_num, y):
        self.X_text = torch.tensor(X_text, dtype=torch.long)  # Text data as long for embedding
        self.X_num = torch.tensor(X_num, dtype=torch.float32)  # Numerical data as float32
        self.y = torch.tensor(y, dtype=torch.long)  # Labels as long for classification

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_text[idx], self.X_num[idx], self.y[idx]

class CrimeCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, max_len, num_num_features):
        super(CrimeCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(30153, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, text, num_features):
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)
        conv_out = self.pool(F.relu(self.conv1(embedded)))
        conv_out = conv_out.view(conv_out.size(0), -1)
        #print(f"conv_out shape: {conv_out.shape}")
        #print(f"num_features shape: {num_features.shape}")
        combined = torch.cat((conv_out, num_features), dim=1)
        #print(f"combined shape: {combined.shape}")
        fc1_out = F.relu(self.fc1(combined))
        out = self.fc2(fc1_out)
        return out

class CNNClassifier:
    def __init__(self, train_data, max_len=100):
        self.train_data = train_data
        self.train_labels = self.train_data['Category']
        self.train_data = self.train_data.drop(['Resolution', 'Category'], axis=1)
        self.max_len = max_len
        
        self.vectorizer = TfidfVectorizer(max_features=10000)
        
        self.le = LabelEncoder()
        self.train_labels = self.le.fit_transform(self.train_labels)
        
        self.train_text_data, self.train_num_data = self.preprocessing(self.train_data)
        
        # Debugging: Check memory usage before splitting
        print(f"Memory Usage before split: {psutil.virtual_memory().percent}%")
        
        # Perform train_test_split separately for text, num, and labels
        X_train_text, X_val_text, y_train, y_val = train_test_split(
            self.train_text_data, self.train_labels, test_size=0.15, random_state=42)
        
        X_train_num, X_val_num = train_test_split(
            self.train_num_data, test_size=0.15, random_state=42)
        
        self.train_dataset = CrimeDataset(X_train_text, X_train_num, y_train)
        self.val_dataset = CrimeDataset(X_val_text, X_val_num, y_val)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=64, shuffle=False)
        
        self.model = CrimeCNN(vocab_size=10000, num_classes=len(self.le.classes_), max_len=max_len, num_num_features=X_train_num.shape[1]).cuda()
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Debugging: Check memory usage after splitting
        print(f"Memory Usage after split: {psutil.virtual_memory().percent}%")
        
    def preprocessing(self, data):
        scaler = StandardScaler()
        num_columns = ["X", "Y", "Year", "Month", "Day", "Hour", "Minute"]
        
        # Extract date features
        data['Dates'] = pd.to_datetime(data['Dates'])
        data['Year'] = data['Dates'].dt.year
        data['Month'] = data['Dates'].dt.month
        data['Day'] = data['Dates'].dt.day
        data['Hour'] = data['Dates'].dt.hour
        data['Minute'] = data['Dates'].dt.minute
        
        # Standardize numerical columns
        data[num_columns] = scaler.fit_transform(data[num_columns])
        
        # Drop unnecessary columns
        data = data.drop(['Dates', 'Address'], axis=1)
        
        # Encode categorical columns
        label_encoder = LabelEncoder()
        data['PdDistrict'] = label_encoder.fit_transform(data['PdDistrict'])
        data['DayOfWeek'] = label_encoder.fit_transform(data['DayOfWeek'])
        
        # Vectorize text data
        descript_features = self.vectorizer.fit_transform(data['Descript']).toarray()
        text_data = descript_features
        
        # Drop the text column after vectorization
        num_data = data.drop(['Descript'], axis=1).values
        
        return text_data, num_data
    
    def train(self, epochs=10):
        for epoch in range(epochs):
            self.model.train()  # Sets the model to training mode
            running_loss = 0.0
            for i, (text, num, labels) in enumerate(self.train_loader):
                text, num, labels = text.cuda(), num.cuda(), labels.cuda()
                self.optimizer.zero_grad()  # Clears the gradients
                outputs = self.model(text, num)  # Forward pass
                loss = self.criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backward pass (compute gradients)
                self.optimizer.step()  # Update the weights
                running_loss += loss.item()
            
            print(f'Epoch {epoch+1}, Loss: {running_loss / len(self.train_loader)}')
            self.validate()  # Validate the model after each epoch
            
    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for text, num, labels in self.val_loader:
                text, num, labels = text.cuda(), num.cuda(), labels.cuda()
                outputs = self.model(text, num)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Validation Accuracy: {100 * correct / total}%')




