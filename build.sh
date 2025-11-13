set -e

mkdir -p data

echo "--- Generating Data (transactions.csv) ---"
python generate_data.py

echo "--- Build Phase complete. Starting API via Gunicorn. ---"