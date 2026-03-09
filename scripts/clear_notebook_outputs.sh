#!/bin/bash
# Clear outputs from all notebooks in the repository
# Usage: bash scripts/clear_notebook_outputs.sh

echo "Clearing notebook outputs..."
echo ""

# Find all .ipynb files and clear their outputs
find notebooks/ -name "*.ipynb" -type f ! -path "*/\.*" | while read notebook; do
    echo "Clearing: $notebook"
    jupyter nbconvert --clear-output --inplace "$notebook"
done

echo ""
echo "✓ All notebook outputs cleared!"
echo ""
echo "To commit these changes:"
echo "  git add notebooks/"
echo "  git commit -m 'Clear notebook outputs'"
echo "  git push origin main"
