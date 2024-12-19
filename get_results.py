from test import *
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HFO Detection')
    parser.add_argument('--test_path', type=str, default=r'/home/test/', help='path to dataset directory')
    parser.add_argument('--save_path', type=str, default='model_conv.pth', help='path to save the model')
    parser.add_argument('--framework', type=str, default="pytorch", help='framework')
    parser.add_argument('--device', type=str, default="cuda:0", help='device')
    args = parser.parse_args()
    evaluation(args.save_path, args.test_path, args.framework, device=args.device)
