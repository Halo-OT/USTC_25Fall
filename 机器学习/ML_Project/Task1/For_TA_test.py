import os
import argparse
import torch
import json
import warnings
from dataset import GlassDefectDataset
from model import ResNet
from utils import load_model
from torch.utils.data import DataLoader

# Disable warnings
warnings.filterwarnings("ignore")

def run_test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize ResNet model
    model = ResNet(num_classes=2).to(device)
    
    # Load weights
    try:
        load_model(model, args.model_path, device)
    except Exception:
        load_model(model, args.model_path)
        
    model.training = False
    
    # Handle path logic
    img_dir = args.test_data_path
    if os.path.exists(os.path.join(args.test_data_path, 'img')):
        img_dir = os.path.join(args.test_data_path, 'img')
    elif os.path.exists(os.path.join(args.test_data_path, 'test', 'img')):
        img_dir = os.path.join(args.test_data_path, 'test', 'img')

    # Dataset setup: reads only images from the directory
    dataset = GlassDefectDataset(img_dir, txt_dir=None, img_size=args.img_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    results = {}
    
    print(f"Starting Inference...")
    print(f" > Data: {img_dir}")
    print(f" > Threshold: {args.threshold}")
    print(f" > TTA: Enabled (6 views)")
    
    with torch.no_grad():
        for imgs, _, names in dataloader:
            imgs = imgs.to(device)
            
            # TTA: 6 views
            scores = torch.softmax(model.forward(imgs), dim=1)
            scores += torch.softmax(model.forward(torch.flip(imgs, dims=[3])), dim=1)
            scores += torch.softmax(model.forward(torch.flip(imgs, dims=[2])), dim=1)
            scores += torch.softmax(model.forward(torch.rot90(imgs, 1, [2, 3])), dim=1)
            scores += torch.softmax(model.forward(torch.rot90(imgs, 2, [2, 3])), dim=1)
            scores += torch.softmax(model.forward(torch.rot90(imgs, 3, [2, 3])), dim=1)
            
            avg_probs = scores / 6.0
            defect_probs = avg_probs[:, 1].cpu().numpy()
            
            for name, prob in zip(names, defect_probs):
                label = True if prob >= args.threshold else False
                results[name] = label
                
    # Output to [student_id].json
    output_filename = f"{args.student_id}.json"
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Prediction result for {args.student_id} saved to {output_filename}")
    print(f"Processed {len(results)} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glass Defect Detection Submission Script")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test data directory")
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to best_model.pth")
    parser.add_argument("--student_id", type=str, default="PB23151782", help="Your Student ID")
    parser.add_argument("--threshold", type=float, default=0.45, help="Decision threshold (default: 0.45)")
    parser.add_argument("--img_size", type=int, default=320, help="Input image size (default: 320)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    
    args = parser.parse_args()
    
    run_test(args)
