import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configurando semente aleatória para reprodutibilidade, usando 42 por conta do Guia do Mochileiro das Galáxias
torch.manual_seed(42)
np.random.seed(42)

# --- Tarefa 1: Carregar os dados ---
print("--- Tarefa 1: Carregando os dados ---")
train_df = pd.read_csv("california_housing_train.csv")
test_df = pd.read_csv("california_housing_test.csv")
print("Dados carregados com sucesso!\n")

# --- Tarefa 2: Exibir informações básicas ---
print("--- Tarefa 2: Informações básicas do dataset ---")
print(f"Dimensões do Treino: {train_df.shape}")
print(f"Dimensões do Teste: {test_df.shape}")
print("\nTipos de dados e Colunas (Conjunto de Treino): ")
print(train_df.dtypes)
print("\nTipos de dados e Colunas (Conjunto de Teste): ")
print(test_df.dtypes)
print("-" * 50 + "\n")

# --- Tarefa 3: Verificar e reportar valores ausentes ---
print("--- Tarefa 3: Verificação de valores ausentes ---")
print("Valores ausentes no Treino:")
print(train_df.isnull().sum())
print("\nValores ausentes no Teste:")
print(test_df.isnull().sum())
print("-" * 50 + "\n")

# --- Tarefa 4: Exibir estatísticas descritivas ---
print("--- Tarefa 4: Estatísticas descritivas ---")
print("Treino: ")
print(train_df.describe())
print("Teste: ")
print(test_df.describe())
print("-" * 50 + "\n")

# --- Tarefa 5: Separar as features (X) da variável alvo (Y) e criar validação ---
print("--- Tarefa 5: Separando Features e Alvo (Treino, Validação, Teste) ---")
target_col = "median_house_value"

X_train_full = train_df.drop(columns=[target_col]).values
y_train_full = train_df[target_col].values

X_test = test_df.drop(columns=[target_col]).values
y_test = test_df[target_col].values

# Divisão do treino original para obter o conjunto de validação (80% treino / 20% validação)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_seed=42
)

print(f"Treino final: {X_train.shape}, Alvo: {y_train.shape}")
print(f"Validação: {X_val.shape}, Alvo: {y_val.shape}")
print(f"Teste: {X_test.shape}, Alvo: {y_test.shape}\n")

# --- Tarefa 6: Aplicar transformação aos dados de entrada ---
print("--- Tarefa 6: Padronizando as features (StandardScaler) ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convertendo os arrays NumPy para Tensores do PyTorch
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Criando DataLoaders para otimizar o treinamento em batches
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
print("Transformação e conversão para tensores concluídas.\n")

# --- Tarefa 7: Construir a arquitetura do MLP ---
print("--- Tarefa 7: Construindo a Arquitetura do MLP ---")    
class CaliforniaHousingMLP(nn.Module):
    def __init__(self, input_dim):
        super(CaliforniaHousingMLP, self).__init__()
        # Arquitetura com duas camadas ocultas, ativação ReLU e Dropout para evitar overfitting
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Saída linear simples para regressão
        )

    def forward(self, x):
        return self.network(x)

input_dim = X_train_scaled.shape[1]
model = CaliforniaHousingMLP(input_dim)
print(model)
print("-" * 50 + "\n")

# --- Tarefa 8: Treinar o modelo utilizando a função de perda RMSE ---
print("--- Tarefa 8: Treinando o Modelo ---")
# Usando o MSE como critério base. A raiz (RMSE) será calculada a cada passo.
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 60
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_X.size(0)
        
    # Cálculo das métricas de época (RMSE)
    epoch_train_rmse = np.sqrt(running_loss / len(train_loader.dataset))
    
    # Avaliação no conjunto de Validação
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor)
        val_mse = criterion(val_preds, y_val_tensor).item()
        epoch_val_rmse = np.sqrt(val_mse)
        
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Época [{epoch+1}/{epochs}] -> Train RMSE: ${epoch_train_rmse:.2f} | Val RMSE: ${epoch_val_rmse:.2f}")

print("\nTreinamento concluído!\n")

# --- Tarefa 9: Fazer previsões no conjunto de teste ---
print("--- Tarefa 9: Fazendo previsões no conjunto de teste ---")
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
print("Previsões geradas com sucesso.\n")

# --- Tarefa 10: Calcular e exibir a métrica RMSE no conjunto de teste ---
print("--- Tarefa 10: Avaliação Final no Conjunto de Teste ---")
test_mse = criterion(test_predictions, y_test_tensor).item()
test_rmse = np.sqrt(test_mse)

print(f"==> RMSE Final no Conjunto de Teste: ${test_rmse:.2f}")

# Amostra visual de comparativo (Real vs Previsto)
print("\nAmostra de Comparações (Primeiras 5 casas):")
for i in range(5):
    print(f"Casa {i+1}: Real: ${y_test[i]:.2f} | Previsto: ${test_predictions[i].item():.2f}")