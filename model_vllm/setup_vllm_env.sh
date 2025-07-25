eval "$(conda shell.bash hook)"
conda create -y -n arc python=3.10
conda activate arc

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r model_vllm/requirements.txt
conda install -y ffmpeg
pip install https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl


export VLLM_PRECOMPILED_WHEEL_LOCATION=$(pwd)/model_vllm/vllm/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
export VLLM_VERSION=v0.8.5.post1-1-gbed41f50d
pip install --editable model_vllm/vllm/
