import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm
import wandb
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherDataset(Dataset):
    def __init__(self, texts, intents, entities=None, preprocessor=None):
        self.texts = texts
        self.intents = intents
        self.entities = entities
        self.preprocessor = preprocessor
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        intent = self.intents[idx]
        
        # Convert text to sequence if preprocessor is available
        if self.preprocessor is not None:
            text = self.preprocessor.text_to_sequence(text)
        
        # Convert to tensors
        text_tensor = torch.tensor(text, dtype=torch.long)
        intent_tensor = torch.tensor(intent, dtype=torch.long)
        
        # Handle entities if available
        if self.entities is not None:
            entity_tensor = torch.tensor(self.entities[idx], dtype=torch.float)
            return {
                'text': text_tensor,
                'intent': intent_tensor,
                'entities': entity_tensor
            }
        
        return {
            'text': text_tensor,
            'intent': intent_tensor
        }

class WeatherTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=1e-3,
        weight_decay=0.01,
        use_wandb=False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss functions
        self.intent_criterion = nn.CrossEntropyLoss()
        self.entity_criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer with different learning rates for different components
        self.optimizer = torch.optim.AdamW([
            {'params': model.embedding.parameters(), 'lr': learning_rate * 0.1},
            {'params': model.lstm.parameters(), 'lr': learning_rate * 0.5},
            {'params': model.intent_classifier.parameters(), 'lr': learning_rate},
            {'params': model.entity_classifier.parameters(), 'lr': learning_rate}
        ], weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.use_wandb = use_wandb
        
        if use_wandb:
            wandb.watch(model)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        intent_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch in progress_bar:
            # Move batch to device
            texts = batch['text'].to(self.device)
            intents = batch['intent'].to(self.device)
            entities = batch.get('entities', None)
            if entities is not None:
                entities = entities.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            intent_logits, entity_logits = self.model(texts)
            
            # Calculate losses
            intent_loss = self.intent_criterion(intent_logits, intents)
            
            if entities is not None:
                entity_loss = self.entity_criterion(entity_logits.mean(dim=1), entities)
                loss = intent_loss + 0.3 * entity_loss
            else:
                loss = intent_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            intent_preds = torch.argmax(intent_logits, dim=1)
            intent_correct += (intent_preds == intents).sum().item()
            total_samples += intents.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{intent_correct/total_samples:.4f}'
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = intent_correct / total_samples
        
        if self.use_wandb:
            wandb.log({
                'train/loss': avg_loss,
                'train/accuracy': accuracy,
                'train/learning_rate': self.optimizer.param_groups[0]['lr']
            })
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        intent_correct = 0
        total_samples = 0
        
        all_intents = []
        all_preds = []
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            # Move batch to device
            texts = batch['text'].to(self.device)
            intents = batch['intent'].to(self.device)
            entities = batch.get('entities', None)
            if entities is not None:
                entities = entities.to(self.device)
            
            # Forward pass
            intent_logits, entity_logits = self.model(texts)
            
            # Calculate losses
            intent_loss = self.intent_criterion(intent_logits, intents)
            
            if entities is not None:
                entity_loss = self.entity_criterion(entity_logits.mean(dim=1), entities)
                loss = intent_loss + 0.3 * entity_loss
            else:
                loss = intent_loss
            
            # Update metrics
            total_loss += loss.item()
            intent_preds = torch.argmax(intent_logits, dim=1)
            intent_correct += (intent_preds == intents).sum().item()
            total_samples += intents.size(0)
            
            # Store predictions for metrics
            all_intents.extend(intents.cpu().numpy())
            all_preds.extend(intent_preds.cpu().numpy())
        
        # Calculate validation metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = intent_correct / total_samples
        
        # Generate classification report
        report = classification_report(
            all_intents,
            all_preds,
            output_dict=True
        )
        
        if self.use_wandb:
            wandb.log({
                'val/loss': avg_loss,
                'val/accuracy': accuracy,
                'val/classification_report': report
            })
            
            # Log confusion matrix
            cm = confusion_matrix(all_intents, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            wandb.log({'val/confusion_matrix': wandb.Image(plt)})
            plt.close()
        
        return avg_loss, accuracy, report
    
    def train(self, num_epochs, save_dir='models'):
        """Train the model for specified number of epochs"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training on device: {self.device}")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            
            # Validate
            val_loss, val_acc, report = self.validate()
            logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_path = save_path / f'best_model_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                }, model_path)
                self.best_model_path = model_path
                logger.info(f"Saved best model to {model_path}")
            
            # Save checkpoint
            checkpoint_path = save_path / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }, checkpoint_path)
        
        logger.info("Training completed!")
        logger.info(f"Best model saved at: {self.best_model_path}")
        
        if self.use_wandb:
            wandb.finish()
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        logger.info(f"Validation loss: {checkpoint['val_loss']:.4f}")
        logger.info(f"Validation accuracy: {checkpoint['val_accuracy']:.4f}")

def prepare_dataloaders(train_path, val_path, preprocessor, batch_size=32, num_workers=4):
    """Prepare training and validation dataloaders"""
    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Load intent mapping
    intent_mapping_path = Path(train_path).parent / 'intent_mapping.json'
    with open(intent_mapping_path, 'r') as f:
        intent_mapping = json.load(f)

    # Map intent strings to integer indices
    train_df['intent'] = train_df['intent'].map(intent_mapping)
    val_df['intent'] = val_df['intent'].map(intent_mapping)

    # Create datasets
    train_dataset = WeatherDataset(
        texts=train_df['text'].values,
        intents=train_df['intent'].values,
        preprocessor=preprocessor
    )
    
    val_dataset = WeatherDataset(
        texts=val_df['text'].values,
        intents=val_df['intent'].values,
        preprocessor=preprocessor
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Example usage
    from backend.data.data_generator import TextPreprocessor
    from backend.models.weather_classifier import create_model
    
    # Load preprocessor
    preprocessor = TextPreprocessor.load_preprocessor('data/preprocessor.json')
    
    # Create model
    model = create_model(
        vocab_size=len(preprocessor.vocab),
        model_type='basic',
        num_intents=6
    )
    
    # Prepare dataloaders
    train_loader, val_loader = prepare_dataloaders(
        train_path='data/train.csv',
        val_path='data/test.csv',
        preprocessor=preprocessor
    )
    
    # Initialize trainer
    trainer = WeatherTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_wandb=True  # Enable wandb logging
    )
    
    # Train model
    trainer.train(num_epochs=20) 