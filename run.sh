set -euo pipefail

RESULT_DIR="results"
CSV_FILE="${RESULT_DIR}/results.csv"
DATASETS=("iris" "adult")

mkdir -p "${RESULT_DIR}"
echo "dataset,loss,accuracy" > "${CSV_FILE}"

for DS in "${DATASETS[@]}"; do
    echo "▶️  Running dataset: ${DS}"
    LOG_FILE="${RESULT_DIR}/${DS}.log"
    echo "    log ➜ ${LOG_FILE}"

    # Run Flower simulation
    flower-simulation \
        --app . \
        --num-supernodes 10 \
        --run-config "dataset=\"${DS}\" num-server-rounds=5 local-epochs=3 penalty=\"l2\"" \
        2>&1 | tee "${LOG_FILE}"

    LOSS=$(grep -E "round 5:" "${LOG_FILE}" | tail -n1 | awk -F': ' '{print $2}') || true
    [[ -z "${LOSS}" ]] && LOSS="NA"

    ACC=$(grep -Eo "'accuracy':[[:space:]]*[0-9.]+" "${LOG_FILE}" | tail -n1 | awk -F': ' '{print $2}') || true
    [[ -z "${ACC}" ]] && ACC="NA"

    # Save results
    echo "${DS},${LOSS},${ACC}" >> "${CSV_FILE}"
done

echo ""
echo "Run completed."
