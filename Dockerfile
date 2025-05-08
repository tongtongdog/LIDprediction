# Base Image
ARG PYTORCH="1.12.0"
ARG CUDA="11.3"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# Environment Variables
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    TZ=Asia/Seoul

# Set Timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# System Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    libgtk2.0-dev \
    python-tk \
    libxrender1 \
    libfontconfig1 \
    sudo \
    vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Python modules
RUN pip install einops jupyterlab medpy monai openpyxl scikit-image scikit-learn scipy==1.7.3 seaborn shap==0.42.1 SimpleITK xgboost==1.6.2 xlsxwriter nibabel

# Set Working Directory
WORKDIR /workspace
