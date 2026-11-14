#!/usr/bin/env bash
# Condor executable: runs `python <args>` from nn/ inside the Vitis AI container,
# with the same environment as `vai` (vai_rc.sh also restores the GPU libs on LD_LIBRARY_PATH)
source /data/gravwav/nmacdona/tools/vai_rc.sh
cd /data/gravwav/nmacdona/deploying-custom-nn-in-fpga/nn || exit 1

# Condor sets CUDA_VISIBLE_DEVICES to GPU UUIDs, but torch 1.13 only parses integer
# indices there and then reports device_count() == 0 (torch.load to cuda fails).
# Translate the UUIDs to NVML indices, which follow PCI bus order.
if [[ $CUDA_VISIBLE_DEVICES == GPU-* ]]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  CUDA_VISIBLE_DEVICES=$(python - <<'EOF'
import ctypes, os
nvml = ctypes.CDLL("libnvidia-ml.so.1")
assert nvml.nvmlInit_v2() == 0, "nvmlInit failed"
indices = []
for uuid in os.environ["CUDA_VISIBLE_DEVICES"].split(","):
    handle, index = ctypes.c_void_p(), ctypes.c_uint()
    assert nvml.nvmlDeviceGetHandleByUUID(uuid.strip().encode(), ctypes.byref(handle)) == 0, uuid
    assert nvml.nvmlDeviceGetIndex(handle, ctypes.byref(index)) == 0, uuid
    indices.append(str(index.value))
print(",".join(indices))
EOF
  ) || exit 1
  export CUDA_VISIBLE_DEVICES
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

exec python -u "$@"
