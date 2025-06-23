# %%
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torchvision.models as models # 사전 학습된 모델을 이용하고자 할 때 사용
from torchvision import transforms, datasets

import torch.nn as nn
import torch.optim as optim

import time
import argparse
from tqdm import tqdm
matplotlib.style.use('ggplot')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# %%
N_h = 100
model = torch.nn.Sequential(
    torch.nn.Linear(1, N_h), 
    torch.nn.ReLU(), 
    torch.nn.Linear(N_h, N_h), 
    torch.nn.ReLU(), 
    torch.nn.Linear(N_h, 1)
) # Dropout x

model_dropout = torch.nn.Sequential(
    torch.nn.Linear(1, N_h), 
    torch.nn.Dropout(0.2), 
    torch.nn.ReLU(), 
    torch.nn.Linear(N_h, N_h), 
    torch.nn.Dropout(0.2), 
    torch.nn.ReLU(), 
    torch.nn.Linear(N_h, 1)
) # Dropout o

train_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(), 
    transforms.RandomVerticalFlip(), 
    transforms.ToTensor(), 
    transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225])
])

# %%
train_dataset = datasets.ImageFolder(
    root = r'/Users/janghyoin/Desktop/2025/Euron/예습 과제/Week14/archive/train', 
    transform = train_transform
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size = 16, shuffle = True
)
val_dataset = datasets.ImageFolder(
    root = r'/Users/janghyoin/Desktop/2025/Euron/예습 과제/Week14/archive/test', 
    transform = val_transform
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size = 16, shuffle = True
)

# %%
def resnet50(pretrained = True, requires_grad = False):
    model = models.resnet50(progress = True, pretrained = pretrained)
    if requires_grad == False: # parameter 고정 -> 역전파 중 기울기 계산 x
        for param in model.parameters():
            param.requires_grad = False
    elif requires_grad == True: # parameter 값이 역전파 중에 기울기 계산에 반영됨
        for param in model.parameters():
            param.requires_grad = True
    model.fc = nn.Linear(2048, 2) # 분류
    return model

# %%
class LRScheduler():
    def __init__(
            self, optimizer, patience = 5, min_lr = 1e-6, factor = 0.5
    ):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode = 'min', 
            patience = self.patience, 
            factor = self.factor, 
            min_lr = self.min_lr, 
            verbose = True
        )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

# %%
class EarlyStopping():
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience        
        self.verbose = verbose
        self.counter = 0
        self.best_score = None          # 검증 데이터셋에 대한 오차 최적화 값(오차가 가장 낮은 값)
        self.early_stop = False         # 조기 종료를 의미하며 초기값은 False로 설정
        self.val_loss_min = np.Inf      # np.Inf(infinity)는 넘파이에서 무한대를 표현
        self.delta = delta              
        self.path = path               

    def __call__(self, val_loss, model):  # 에포크만큼 학습이 반복되면서 best_loss가 갱신되고,
                                          # best_loss에 진전이 없으면 조기 종료 후 모델을 저장
        score = -val_loss

        if self.best_score is None:       # best_score에 값이 존재하지 않으면 실행
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:  # best_score + delta가 score보다 크면 실행
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:                             # 그 외 모든 경우에 실행
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
 
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f}) --> {val_loss:.6f}')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--lr-scheduler', dest = 'lr_scheduler', action = 'store_true')
parser.add_argument('--early-stopping', dest = 'early_stopping', action = 'store_true')
args, unknown = parser.parse_known_args()
args = vars(args)

# %%
print(f"Computation device: {device}\n")
model = models.resnet50(pretrained = True).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params} total parameters.")

total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
print(f"{total_trainable_params} training parameters.")

# %%
lr = 0.001
epochs = 10
optimizer = torch.optim.Adam(model_dropout.parameters(), lr = 0.01)
criterion = nn.CrossEntropyLoss()

# %%
loss_plot_name = 'loss' # 오차 출력에 대한 문자열
acc_plot_name = 'accuracy' # 정확도 출력에 대한 문자열
model_name = 'model' # 모델을 지정하기 위한 문자열

# %%
if args['lr_scheduler']:
    print('INFO: Initializing learning rate scheduler')
    lr_scheduler = LRScheduler(optimizer)
    loss_plot_name = 'lrs_loss'
    acc_plot_name = 'lrs_accuracy'
    model_name = 'lrs_model'
if args['early_stopping']:
    print('INFO: Initializing early stopping')
    early_stopping = EarlyStopping()
    loss_plot_name = 'es_loss'
    acc_plot_name = 'es_accuracy'
    model_name = 'es_model'

# %%
def training(model, train_dataloader, train_dataset, optimizer, criterion):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(train_dataloader), 
                    total=int(len(train_dataset)/train_dataloader.batch_size))  # 훈련 진행 과정을 시각적으로 표현

    for i, data in prog_bar:
        counter += 1
        data, target = data[0].to(device), data[1].to(device)
        total += target.size(0)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / counter
    train_accuracy = 100. * train_running_correct / total
    return train_loss, train_accuracy


# %%
def validate(model, test_dataloader, val_dataset, criterion):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(test_dataloader), 
                    total=int(len(val_dataset)/test_dataloader.batch_size))  # 모델 검증 과정을 시각적으로 표현

    with torch.no_grad():
        for i, data in prog_bar:
            counter += 1
            data, target = data[0].to(device), data[1].to(device)
            total += target.size(0)
            outputs = model(data)
            loss = criterion(outputs, target)

            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()

    val_loss = val_running_loss / counter
    val_accuracy = 100. * val_running_correct / total
    return val_loss, val_accuracy


# %%
train_loss, train_accuracy = [], []  # 훈련 데이터셋을 이용한 결과 저장 리스트
val_loss, val_accuracy = [], []      # 검증 데이터셋을 이용한 결과 저장 리스트

start = time.time()
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    
    train_epoch_loss, train_epoch_accuracy = training(
        model, train_dataloader, train_dataset, optimizer, criterion
    )

    val_epoch_loss, val_epoch_accuracy = validate(
        model, val_dataloader, val_dataset, criterion
    )

    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

    if args['lr_scheduler']:  # 인수 값이 lr_scheduler이면 실행
        lr_scheduler(val_epoch_loss)

    if args['early_stopping']:  # 인수 값이 early_stopping이면 실행
        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop:
            break

    print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}")
    print(f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}")

end = time.time()
print(f"Training time: {(end-start)/60:.3f} minutes")


# %%
print('Saving loss and accuracy plots...')

plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='green', label='train accuracy')  # 훈련 데이터셋에 대한 정확도를 그래프로 출력
plt.plot(val_accuracy, color='blue', label='validation accuracy')  # 검증 데이터셋에 대한 정확도를 그래프로 출력
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f"img/{acc_plot_name}.png")
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')  # 훈련 데이터셋에 대한 오차를 그래프로 출력
plt.plot(val_loss, color='red', label='validation loss')  # 검증 데이터셋에 대한 오차를 그래프로 출력
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"img/{loss_plot_name}.png")
plt.show()

print('Saving model...')
torch.save(model.state_dict(), f"img/{model_name}.pth")  # 모델을 저장
print('TRAINING COMPLETE')



