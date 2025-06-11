import argparse
import logging
from pathlib import Path
import wandb
from data.data_generator import WeatherDatasetGenerator, TextPreprocessor
from models.weather_classifier import create_model
from training.trainer import WeatherTrainer, prepare_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Weather Intent Classifier')
    
    # Data generation arguments
    parser.add_argument('--generate-data', action='store_true',
                      help='Generate new training data')
    parser.add_argument('--samples-per-intent', type=int, default=1000,
                      help='Number of samples to generate per intent')
    
    # Model arguments
    parser.add_argument('--model-type', choices=['basic', 'advanced'], default='basic',
                      help='Type of model to use')
    parser.add_argument('--embed-dim', type=int, default=128,
                      help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=256,
                      help='Hidden dimension for LSTM')
    parser.add_argument('--dropout', type=float, default=0.3,
                      help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=20,
                      help='Number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                      help='Weight decay for optimizer')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of workers for data loading')
    
    # Wandb arguments
    parser.add_argument('--use-wandb', action='store_true',
                      help='Use Weights & Biases for logging')
    parser.add_argument('--wandb-project', type=str, default='weather-intent-classifier',
                      help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                      help='Wandb entity name')
    
    # Path arguments
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Directory for data files')
    parser.add_argument('--model-dir', type=str, default='models',
                      help='Directory for model checkpoints')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create directories
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    data_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    
    # Generate data if requested
    if args.generate_data:
        logger.info("Generating training data...")
        generator = WeatherDatasetGenerator()
        df = generator.generate_training_data(samples_per_intent=args.samples_per_intent)
        generator.save_dataset(df, output_dir=data_dir)
        
        # Build and save preprocessor
        preprocessor = TextPreprocessor()
        preprocessor.build_vocab(df['text'])
        preprocessor.save_preprocessor(data_dir / 'preprocessor.json')
    else:
        # Load existing preprocessor
        preprocessor = TextPreprocessor.load_preprocessor(data_dir / 'preprocessor.json')
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args)
        )
    
    # Create model
    model = create_model(
        vocab_size=len(preprocessor.vocab),
        model_type=args.model_type,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    
    # Prepare dataloaders
    train_loader, val_loader = prepare_dataloaders(
        train_path=data_dir / 'train.csv',
        val_path=data_dir / 'test.csv',
        preprocessor=preprocessor,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Initialize trainer
    trainer = WeatherTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_wandb=args.use_wandb
    )
    
    # Train model
    trainer.train(
        num_epochs=args.num_epochs,
        save_dir=model_dir
    )

if __name__ == '__main__':
    main() 