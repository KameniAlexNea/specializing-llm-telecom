python -m pip install "torch==2.1.1" torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu118
python -m pip install transformers[torch] --extra-index https://download.pytorch.org/whl/cu118
python -m pip install "unsloth[cu118-torch211] @ git+https://github.com/unslothai/unsloth.git" --extra-index https://download.pytorch.org/whl/cu118
python -m pip install trl peft accelerate bitsandbytes --extra-index https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt --extra-index https://download.pytorch.org/whl/cu118
# --no-deps 
# sudo apt install libcusparse-11-8 if necessary