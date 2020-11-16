# To install PyTorch on macOS:
#   pip3 install torch==1.5.1 torchvision==0.6.1

# To install PyTorch on Linux:
#   pip3 install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

pip3 freeze | grep -vwE "(torch|torchvision)" > requirements.txt
