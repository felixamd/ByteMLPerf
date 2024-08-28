FROM rocm/pytorch:rocm6.2_ubuntu20.04_py3.9_pytorch_release_2.1.2

RUN mkdir /src && cd /src
WORKDIR /src

# install migraphx
RUN pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz

RUN git clone https://github.com/ROCm/AMDMIGraphX.git && cd AMDMIGraphX &&  rbuild build -d depend -B build -DGPU_TARGETS=gfx942 && cd build && make -j install

# install onnx
RUN git clone https://github.com/microsoft/onnxruntime.git && sed -i 's|cd /onnxruntime|cd /src/onnxruntime|' /src/AMDMIGraphX/tools/build_and_test_onnxrt.sh && sed -i '/pai_test_launcher.sh/s/^/#/' /src/AMDMIGraphX/tools/build_and_test_onnxrt.sh && /src/AMDMIGraphX/tools/build_and_test_onnxrt.sh

RUN pip install /src/onnxruntime/build/Linux/Release/dist/*.whl

# download ByteMLPerf
RUN git clone https://github.com/anghostcici/ByteMLPerf.git -b migraphx_backend

RUN pip install tensorflow tf2onnx bert-tensorflow==1.0.1
