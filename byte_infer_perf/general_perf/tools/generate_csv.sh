#!/bin/bash

TRANSPOSE=False

ORI_CSV=/tmp/ori.csv
DATE_=$(date +%m-%d_%H-%M)
OUT_DIR=$2/${DATE_}

mkdir -p ${OUT_DIR}

if [[ "$1" == "32" ]]; then
        DTYPE="fp32"
else
        DTYPE="fp16"
fi


for MODEL in unet clip vae-encoder vae-decoder
do
        echo " ------ model ${MODEL} ------ "
        jq -r '.Performance[] | [.BS, .QPS, ."AVG Latency", ."P99 Latency"] | @csv' reports/MIGRAPHX/${MODEL}-onnx-${DTYPE}/result-${DTYPE}.json > $ORI_CSV
        jq -r '.Accuracy | [."Mean Diff", ."Std Diff", ."Max Diff", ."Max Rel-Diff", ."Mean Rel-Diff", ."Diff Dist"] | @csv' reports/MIGRAPHX/${MODEL}-onnx-fp32/result-${DTYPE}.json >> $ORI_CSV
        if [[ "$TRANSPOSE" == "True" ]]; then
                csvtool transpose $ORI_CSV > ${OUT_DIR}/${MODEL}-${DTYPE}.csv
                csvtool transpose $ORI_CSV >> ${OUT_DIR}/stable_diffusion_${DTYPE}_${DATE_}.csv
        else
                cat $ORI_CSV >> ${OUT_DIR}/stable_diffusion_${DTYPE}_${DATE_}.csv
                cp reports/MIGRAPHX/${MODEL}-onnx-${DTYPE}/result-${DTYPE}.json ${OUT_DIR}/result-${DTYPE}-${MODEL}.json
        fi
        echo "${OUT_DIR}/result-${DTYPE}-${MODEL}.json saved."
	report_name=${MODEL}-onnx-fp32-to-${DTYPE}
	report_name_upper=${report_name^^}
	cp reports/MIGRAPHX/${MODEL}-onnx-${DTYPE}/${report_name}.png ${OUT_DIR}
	cp reports/MIGRAPHX/${MODEL}-onnx-${DTYPE}/${report_name_upper}.JSON.pdf ${OUT_DIR}
        echo "${OUT_DIR}/${MODEL}-onnx-fp32-to-${DTYPE}.png saved"
done

echo " ------ ALL ------ "
echo "${OUT_DIR}/stable_diffusion_${DTYPE}_${DATE_}.csv saved."
