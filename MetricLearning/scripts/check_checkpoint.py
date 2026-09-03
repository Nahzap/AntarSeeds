"""
Script para verificar contenido del checkpoint
"""
import torch
import sys

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "models/final_model.pth"

print(f"Verificando checkpoint: {checkpoint_path}")
print("=" * 70)

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("\nClaves en el checkpoint:")
for key in checkpoint.keys():
    value = checkpoint[key]
    if isinstance(value, torch.Tensor):
        print(f"  {key}: Tensor {value.shape}")
    elif isinstance(value, dict):
        print(f"  {key}: Dict con {len(value)} items")
    elif isinstance(value, list):
        print(f"  {key}: List con {len(value)} items")
    else:
        print(f"  {key}: {type(value).__name__}")

print("\n" + "=" * 70)

# Verificar si tiene embeddings de referencia
if 'reference_embeddings' in checkpoint:
    print("✅ Checkpoint TIENE embeddings de referencia")
    print(f"   - Embeddings: {checkpoint['reference_embeddings'].shape}")
    print(f"   - Labels: {checkpoint['reference_labels'].shape}")
    print(f"   - Clases: {checkpoint['class_names']}")
else:
    print("❌ Checkpoint NO TIENE embeddings de referencia")
    print("\nSoluciones:")
    print("  1. Reentrenar: python app.py --cli train --config config.yaml")
    print("  2. Usar base de datos temporal: proporcionar data/processed/train")
