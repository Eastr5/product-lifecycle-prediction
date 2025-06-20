# src/models/autoencoder.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import logging
import os
import joblib
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional, Any

class AutoencoderDataset(Dataset):
    """
    用于自编码器训练的 PyTorch 数据集。
    输入和目标都是相同的数值特征。
    """
    def __init__(self, data_tensor: torch.Tensor):
        super().__init__()
        self.data_tensor = data_tensor
        logging.info(f"AutoencoderDataset initialized with data shape: {data_tensor.shape}")

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.data_tensor[idx]

class Autoencoder(nn.Module):
    """
    标准的深度自编码器模型。
    """
    def __init__(self, 
                 input_dim: int, 
                 encoding_dim: int, 
                 encoder_layers_config: List[int], 
                 decoder_layers_config: List[int], 
                 activation_fn_str: str = "ReLU"):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        if activation_fn_str.lower() == "relu":
            activation_fn = nn.ReLU()
        elif activation_fn_str.lower() == "leakyrelu":
            activation_fn = nn.LeakyReLU()
        elif activation_fn_str.lower() == "tanh":
            activation_fn = nn.Tanh()
        else:
            logging.warning(f"Unsupported activation function '{activation_fn_str}'. Defaulting to ReLU.")
            activation_fn = nn.ReLU()

        # --- 构建编码器 ---
        encoder_modules = []
        current_dim = input_dim
        if not encoder_layers_config: 
            encoder_modules.append(nn.Linear(current_dim, encoding_dim))
            encoder_modules.append(activation_fn) 
        else:
            for layer_units in encoder_layers_config:
                if layer_units <= 0:
                    logging.warning(f"Encoder layer with {layer_units} units is invalid. Skipping.")
                    continue
                encoder_modules.append(nn.Linear(current_dim, layer_units))
                encoder_modules.append(activation_fn)
                current_dim = layer_units
            encoder_modules.append(nn.Linear(current_dim, encoding_dim))
            encoder_modules.append(activation_fn) 

        self.encoder = nn.Sequential(*encoder_modules)
        logging.info(f"Encoder built: {self.encoder}")

        # --- 构建解码器 ---
        decoder_modules = []
        current_dim = encoding_dim
        if not decoder_layers_config: 
            decoder_modules.append(nn.Linear(current_dim, input_dim))
        else:
            for layer_units in decoder_layers_config:
                if layer_units <= 0:
                    logging.warning(f"Decoder layer with {layer_units} units is invalid. Skipping.")
                    continue
                decoder_modules.append(nn.Linear(current_dim, layer_units))
                decoder_modules.append(activation_fn)
                current_dim = layer_units
            decoder_modules.append(nn.Linear(current_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_modules)
        logging.info(f"Decoder built: {self.decoder}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Helper method to get only the encoded representation."""
        return self.encoder(x)


def train_autoencoder(
    features_df: pd.DataFrame,
    numeric_cols_to_encode: List[str],
    ae_config: Dict[str, Any],
    device: torch.device # device is passed here
) -> Tuple[Optional[nn.Sequential], Optional[StandardScaler]]:
    """
    训练自编码器模型。
    """
    logging.info(f"Starting Autoencoder training for {len(numeric_cols_to_encode)} features on device: {device}")
    if not numeric_cols_to_encode:
        logging.error("No numeric columns specified for autoencoder training.")
        return None, None
    
    data_to_encode = features_df[numeric_cols_to_encode].copy()
    if data_to_encode.isnull().any().any():
        logging.warning("NaN values found in features for autoencoder. Filling with 0.")
        data_to_encode = data_to_encode.fillna(0)
    
    scaler = StandardScaler()
    try:
        scaled_data = scaler.fit_transform(data_to_encode.values)
    except ValueError as e:
        logging.error(f"Error during scaling data for autoencoder: {e}. Check for non-numeric data or all-zero columns.")
        return None, None
        
    data_tensor = torch.FloatTensor(scaled_data) # Data tensor initially on CPU
    
    dataset = AutoencoderDataset(data_tensor) # Dataset holds CPU tensor
    if len(dataset) == 0:
        logging.error("AutoencoderDataset is empty. Cannot train autoencoder.")
        return None, scaler

    batch_size = ae_config.get('batch_size', 64)
    # DataLoader will fetch CPU tensors, they are moved to device in the loop
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=ae_config.get('num_workers', 0))

    input_dim = data_tensor.shape[1]
    encoding_dim = ae_config.get('encoding_dim', 16)
    encoder_layers = ae_config.get('encoder_layers', [64, 32])
    decoder_layers = ae_config.get('decoder_layers', [32, 64]) 
    activation_fn_str = ae_config.get('activation_function', 'ReLU')

    model = Autoencoder(
        input_dim=input_dim,
        encoding_dim=encoding_dim,
        encoder_layers_config=encoder_layers,
        decoder_layers_config=decoder_layers,
        activation_fn_str=activation_fn_str
    ).to(device) # Move the entire model to the target device

    criterion = nn.MSELoss()
    optimizer_name = ae_config.get('optimizer', 'Adam')
    learning_rate = ae_config.get('learning_rate', 0.001)

    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        logging.warning(f"Unsupported optimizer '{optimizer_name}'. Defaulting to Adam.")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epochs = ae_config.get('epochs', 50)
    patience = ae_config.get('early_stopping_patience', 10)
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state_encoder = None
    best_model_state_autoencoder = None # For saving the full AE if needed

    logging.info(f"Training Autoencoder for {epochs} epochs with encoding_dim={encoding_dim} on device {device}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader: 
            inputs, targets = inputs.to(device), targets.to(device) # Move batch data to device
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        logging.info(f"AE Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            # Get state_dict from model.encoder which is already on the correct device
            best_model_state_encoder = model.encoder.state_dict() 
            best_model_state_autoencoder = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logging.info(f"Autoencoder early stopping triggered after {epoch+1} epochs.")
            break
            
    encoder_model_path = ae_config.get('encoder_model_path')
    autoencoder_model_path = ae_config.get('autoencoder_model_path')
    scaler_path = ae_config.get('scaler_path')

    final_encoder_to_return = None # Initialize
    if best_model_state_encoder:
        # Reconstruct the encoder structure and load the best state.
        # The encoder will be on the same device as the 'model' from which 'best_model_state_encoder' was derived.
        final_encoder_to_return = Autoencoder(input_dim, encoding_dim, encoder_layers, [], activation_fn_str).encoder
        final_encoder_to_return.load_state_dict(best_model_state_encoder)
        final_encoder_to_return.to(device) # Explicitly ensure it's on the target device

        if encoder_model_path:
            os.makedirs(os.path.dirname(encoder_model_path), exist_ok=True)
            torch.save(final_encoder_to_return.state_dict(), encoder_model_path)
            logging.info(f"Trained encoder saved to {encoder_model_path}")
        
        if autoencoder_model_path and best_model_state_autoencoder:
            os.makedirs(os.path.dirname(autoencoder_model_path), exist_ok=True)
            # Save the full AE state dict, which will also be on the correct device
            torch.save(best_model_state_autoencoder, autoencoder_model_path) 
            logging.info(f"Trained full autoencoder saved to {autoencoder_model_path}")
    else:
        logging.warning("No best model state found for encoder. Returning None.")
        return None, scaler

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logging.info(f"Scaler for autoencoder input saved to {scaler_path}")

    return final_encoder_to_return, scaler # final_encoder_to_return is now guaranteed to be on 'device'

if __name__ == '__main__':
    # ... (rest of the __main__ block for testing, unchanged) ...
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    mock_features_df = pd.DataFrame(np.random.rand(100, 10), columns=[f'feat_{i}' for i in range(10)])
    mock_numeric_cols = [f'feat_{i}' for i in range(10)]
    
    mock_ae_config = {
        'encoding_dim': 3,
        'encoder_layers': [8, 5], 
        'decoder_layers': [5, 8], 
        'activation_function': 'ReLU',
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'epochs': 5, 
        'batch_size': 16,
        'early_stopping_patience': 3,
        'scaler_path': 'temp_ae_scaler.pkl',
        'encoder_model_path': 'temp_trained_encoder.pt',
        'autoencoder_model_path': 'temp_trained_autoencoder.pt',
        'num_workers': 0
    }
    # Determine device for testing (prefer CUDA if available)
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"--- Autoencoder __main__ test running on device: {test_device} ---")


    print("--- Testing Autoencoder Training ---")
    trained_encoder, used_scaler = train_autoencoder(
        mock_features_df,
        mock_numeric_cols,
        mock_ae_config,
        test_device # Pass the determined device
    )

    if trained_encoder and used_scaler:
        print("Autoencoder training test successful.")
        # The trained_encoder is already on test_device
        print(f"Trained Encoder (on {next(trained_encoder.parameters()).device}):\n{trained_encoder}")


        sample_data = mock_features_df[mock_numeric_cols].head().values
        scaled_sample = used_scaler.transform(sample_data)
        sample_tensor = torch.FloatTensor(scaled_sample).to(test_device) # Move sample to test_device
        
        trained_encoder.eval() 
        with torch.no_grad():
            encoded_output = trained_encoder(sample_tensor)
        print(f"Sample original data shape: {sample_tensor.shape}")
        print(f"Encoded output shape: {encoded_output.shape}, Device: {encoded_output.device}")
        assert encoded_output.shape[1] == mock_ae_config['encoding_dim']
        assert encoded_output.device == test_device 
        print("Encoding test successful.")

        for p in [mock_ae_config['scaler_path'], mock_ae_config['encoder_model_path'], mock_ae_config['autoencoder_model_path']]:
            if os.path.exists(p):
                os.remove(p)
    else:
        print("Autoencoder training test failed.")
