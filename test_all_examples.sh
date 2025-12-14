#!/bin/bash
# Test all example scripts and save results

set -e  # Exit on error for critical sections

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Create output directory for test results
OUTPUT_DIR="output/test_results"
mkdir -p "$OUTPUT_DIR"

# Create log file
LOG_FILE="$OUTPUT_DIR/test_log_$(date +%Y%m%d_%H%M%S).txt"
RESULTS_FILE="$OUTPUT_DIR/test_results_$(date +%Y%m%d_%H%M%S).json"

echo "========================================="
echo "Testing All Example Scripts"
echo "========================================="
echo "Log file: $LOG_FILE"
echo "Results: $RESULTS_FILE"
echo ""

# Activate virtual environment
if [ ! -d "venv_test" ]; then
    echo "Error: Virtual environment not found. Creating..."
    python3 -m venv venv_test
    source venv_test/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
else
    source venv_test/bin/activate
fi

echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'Not installed')"
echo ""

# Find all numbered example scripts
EXAMPLE_SCRIPTS=$(find examples -name "[0-9]*.sh" -type f | sort)

if [ -z "$EXAMPLE_SCRIPTS" ]; then
    echo "Error: No example scripts found"
    exit 1
fi

echo "Found $(echo "$EXAMPLE_SCRIPTS" | wc -l | tr -d ' ') example script(s)"
echo ""

# Initialize results tracking
PASSED=0
FAILED=0
SKIPPED=0
FAILED_SCRIPTS=()
PASSED_SCRIPTS=()
SKIPPED_SCRIPTS=()

# Test each script
for script in $EXAMPLE_SCRIPTS; do
    script_name=$(basename "$script")
    echo "----------------------------------------"
    echo "Testing: $script_name"
    echo "----------------------------------------"
    
    # Skip script 08 (quantize_trained_model) if no trained model exists
    if [[ "$script_name" == "08_quantize_trained_model.sh" ]]; then
        TRAINED_MODEL="output/examples/basic_single_head/person_detection_binary.keras"
        if [ ! -f "$TRAINED_MODEL" ]; then
            echo "  ⏭️  SKIPPED: No trained model found (this is expected)"
            echo "      To test this, first run: bash examples/01_basic_single_head.sh"
            ((SKIPPED++))
            SKIPPED_SCRIPTS+=("$script_name")
            continue
        fi
    fi
    
    # Run the script with timeout
    start_time=$(date +%s)
    
    if bash "$script" > "$OUTPUT_DIR/${script_name}.log" 2>&1; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "  ✅ PASSED (${duration}s)"
        ((PASSED++))
        PASSED_SCRIPTS+=("$script_name")
        
        # Check if output files were created
        if grep -q "int8.tflite" "$OUTPUT_DIR/${script_name}.log"; then
            echo "    ✓ TFLite model created"
        fi
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "  ❌ FAILED (${duration}s)"
        echo "    Check log: $OUTPUT_DIR/${script_name}.log"
        ((FAILED++))
        FAILED_SCRIPTS+=("$script_name")
        
        # Show last few lines of error
        echo "    Last error:"
        tail -n 3 "$OUTPUT_DIR/${script_name}.log" | sed 's/^/      /'
    fi
    echo ""
done

# Create summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Total scripts: $((PASSED + FAILED + SKIPPED))"
echo "✅ Passed: $PASSED"
echo "❌ Failed: $FAILED"
echo "⏭️  Skipped: $SKIPPED"
echo ""

if [ ${#PASSED_SCRIPTS[@]} -gt 0 ]; then
    echo "Passed scripts:"
    for s in "${PASSED_SCRIPTS[@]}"; do
        echo "  ✅ $s"
    done
    echo ""
fi

if [ ${#FAILED_SCRIPTS[@]} -gt 0 ]; then
    echo "Failed scripts:"
    for s in "${FAILED_SCRIPTS[@]}"; do
        echo "  ❌ $s"
    done
    echo ""
fi

if [ ${#SKIPPED_SCRIPTS[@]} -gt 0 ]; then
    echo "Skipped scripts:"
    for s in "${SKIPPED_SCRIPTS[@]}"; do
        echo "  ⏭️  $s"
    done
    echo ""
fi

# Save results to JSON file
cat > "$RESULTS_FILE" <<EOF
{
  "test_date": "$(date -Iseconds)",
  "total": $((PASSED + FAILED + SKIPPED)),
  "passed": $PASSED,
  "failed": $FAILED,
  "skipped": $SKIPPED,
  "passed_scripts": $(printf '%s\n' "${PASSED_SCRIPTS[@]}" | jq -R . | jq -s .),
  "failed_scripts": $(printf '%s\n' "${FAILED_SCRIPTS[@]}" | jq -R . | jq -s .),
  "skipped_scripts": $(printf '%s\n' "${SKIPPED_SCRIPTS[@]}" | jq -R . | jq -s .),
  "python_version": "$(python --version 2>&1)",
  "tensorflow_version": "$(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'N/A')"
}
EOF

echo "Results saved to: $RESULTS_FILE"
echo "All logs saved to: $OUTPUT_DIR/"
echo ""

# List created models
echo "Generated models:"
find output/examples -name "*.tflite" 2>/dev/null | while read -r model; do
    size=$(du -h "$model" | cut -f1)
    echo "  📦 $model ($size)"
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "🎉 All tests passed!"
    exit 0
else
    echo "⚠️  Some tests failed. Check logs in $OUTPUT_DIR/"
    exit 1
fi

