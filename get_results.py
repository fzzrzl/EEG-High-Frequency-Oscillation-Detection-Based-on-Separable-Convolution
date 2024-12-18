from test import *

if __name__ == "__main__":
    model_path = "model_conv_kernel17_9459.pth"
    test_data_dir = "test/"
    framework = "pytorch"
    evaluation(model_path, test_data_dir, framework, device="cuda:0")