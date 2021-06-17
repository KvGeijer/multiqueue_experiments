#!/bin/bash

set -uo pipefail

hostname=$(cat /proc/sys/kernel/hostname)
bin_dir="./build_${hostname}"
reps=5
timeout=3

name="$1"

experiment_dir="${HOME}/experiments/${hostname}/multiqueue"
log_dir="${experiment_dir}/logs/${name}"
result_dir="${experiment_dir}/results/${name}"
quality_bin="${bin_dir}/benchmarks/${name}_quality"
throughput_bin="${bin_dir}/benchmarks/${name}_throughput"

echo Quality bin: ${quality_bin} >&2
echo Throughput bin: ${throughput_bin} >&2

mkdir -p ${log_dir}
mkdir -p ${result_dir}

if [[ -x ${quality_bin} ]]; then
    echo Starting quality benchmark >&2
    for ((j=1;j<=$(nproc);j=2*j)); do
        if [[ -f  "${log_dir}/evaluate_quality_${j}_stderr.txt" ]]; then
            echo Quality benchmark exists, skipping... >&2
        else
            echo "${quality_bin} -j ${j}" >&2
            ${quality_bin} -j ${j} 2> "${log_dir}/quality_${j}_stderr.txt" | ${bin_dir}/utils/evaluate_quality -r "${result_dir}/rank_${j}.txt" -d "${result_dir}/delay_${j}.txt" -t "${result_dir}/top_delay_${j}.txt" 2> "${log_dir}/evaluate_quality_${j}_stderr.txt"
        fi
    done
else
    echo Quality binary not executable, skipping >&2
fi

if [[ -x ${throughput_bin} ]]; then
    if [[ -f "${result_dir}/throughput.txt" ]]; then
        echo Throughput benchmark exists, skipping... >&2
    else
        echo Starting throughput benchmarks >&2
        {
            echo "threads rep throughput"
            for ((j=1;j<=$(nproc);j=2*j)); do
                for r in $(seq 1 $reps);do
                    echo "${throughput_bin} -j ${j} -t ${timeout}000" >&2
                    ${throughput_bin} -j ${j} -t ${timeout}000 2> "${log_dir}/throughput_${j}_${r}_stderr.txt" >  "${log_dir}/throughput_${j}_${r}_stdout.txt"
                    ops=$(grep "Ops/s:" "${log_dir}/throughput_${j}_${r}_stdout.txt" | cut -d' ' -f2)
                    echo $j $r $ops
                done
            done
        } > "${result_dir}/throughput.txt"
    fi
else
    echo Throughput binary not executable, skipping >&2
fi
