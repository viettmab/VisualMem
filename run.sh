#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Common parameters for all scripts
# LIB="memos"
# VERSION="default"
LIB="ours"
VERSION="window-full"
WORKERS=10
TOPK=20

python -m evaluation.ingestion --lib $LIB --version $VERSION --workers $WORKERS --window-size full
if [ $? -ne 0 ]; then
    echo "Error running ingestion"
    exit 1
fi

python -m evaluation.search --lib $LIB --version $VERSION --top_k $TOPK --workers $WORKERS
if [ $? -ne 0 ]; then
    echo "Error running search"
    exit 1
fi

python -m evaluation.responses --lib $LIB --version $VERSION --workers $WORKERS
python -m evaluation.statistics --lib $LIB --version $VERSION

echo "All scripts completed successfully!"