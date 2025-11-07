#!/bin/bash
set -e

ENSEMBLE_FILE="/home/leaf/Pyrofex/nunet/device-management-service/standalone-demo/ensemble.yml"
RNODE_DEPLOY="./node/target/universal/stage/bin/rnode deploy"
RNODE_PROPOSE="./node/target/universal/stage/bin/rnode propose"
PRIVATE_KEY="1111111111111111111111111111111111111111111111111111111111111111"
POLL_INTERVAL=5
MAX_POLLS=60

echo "==================================================="
echo "Nunet DMS Deployment Test Script"
echo "==================================================="
echo ""

# Check that ensemble file exists
if [ ! -f "$ENSEMBLE_FILE" ]; then
    echo "ERROR: Ensemble file not found: $ENSEMBLE_FILE"
    exit 1
fi

echo "Step 1: Deploying ensemble..."
echo "---------------------------------------------------"
$RNODE_DEPLOY --private-key "$PRIVATE_KEY" --phlo-limit 1000000 --phlo-price 1 test-nunet-deploy.rho
echo ""
echo "Creating block..."
PROPOSE_OUTPUT=$($RNODE_PROPOSE 2>&1)
echo "$PROPOSE_OUTPUT"
echo ""

# Wait a moment for the contract to execute
sleep 2

# Now get the ensemble ID by querying
echo "Querying for ensemble ID..."
$RNODE_DEPLOY --private-key "$PRIVATE_KEY" --phlo-limit 1000000 --phlo-price 1 test-nunet-get-id.rho
echo ""
echo "Creating block..."
$RNODE_PROPOSE
sleep 1

# Extract ensemble ID from the output (this is a simplified approach)
# In reality, we'd need to query the registry or use a return channel
ENSEMBLE_ID=$(echo "$PROPOSE_OUTPUT" | grep -oP '(?<=Ensemble ID: )[a-f0-9]{64}' | head -1)

if [ -z "$ENSEMBLE_ID" ]; then
    echo "ERROR: Failed to extract ensemble ID from deployment output"
    echo "Output was:"
    echo "$DEPLOY_OUTPUT"
    exit 1
fi

echo "Extracted Ensemble ID: $ENSEMBLE_ID"
echo ""

echo "Step 2: Polling deployment status..."
echo "---------------------------------------------------"

POLL_COUNT=0
FINAL_STATUS=""

while [ $POLL_COUNT -lt $MAX_POLLS ]; do
    echo "Poll $((POLL_COUNT + 1))/$MAX_POLLS..."

    STATUS_OUTPUT=$($RNODE_EVAL test-nunet-status.rho 2>&1)
    echo "$STATUS_OUTPUT"

    # Check if status contains "Completed" or "Failed"
    if echo "$STATUS_OUTPUT" | grep -q "Completed"; then
        FINAL_STATUS="Completed"
        echo ""
        echo "✓ Deployment completed successfully!"
        break
    elif echo "$STATUS_OUTPUT" | grep -q "Failed"; then
        FINAL_STATUS="Failed"
        echo ""
        echo "✗ Deployment failed!"
        break
    fi

    POLL_COUNT=$((POLL_COUNT + 1))

    if [ $POLL_COUNT -lt $MAX_POLLS ]; then
        echo "Status not final yet, waiting ${POLL_INTERVAL}s..."
        sleep $POLL_INTERVAL
    fi
    echo ""
done

if [ -z "$FINAL_STATUS" ]; then
    echo "WARNING: Deployment status polling timed out after $((MAX_POLLS * POLL_INTERVAL)) seconds"
    echo ""
fi

echo "Step 3: Retrieving deployment manifest..."
echo "---------------------------------------------------"
MANIFEST_OUTPUT=$($RNODE_EVAL test-nunet-manifest.rho 2>&1)
echo "$MANIFEST_OUTPUT"
echo ""

echo "Step 4: Retrieving deployment logs..."
echo "---------------------------------------------------"
LOGS_OUTPUT=$($RNODE_EVAL test-nunet-logs.rho 2>&1)
echo "$LOGS_OUTPUT"
echo ""

echo "==================================================="
echo "Deployment Test Complete"
echo "==================================================="
echo "Ensemble ID: $ENSEMBLE_ID"
echo "Final Status: ${FINAL_STATUS:-Unknown/Timeout}"
echo ""

exit 0
