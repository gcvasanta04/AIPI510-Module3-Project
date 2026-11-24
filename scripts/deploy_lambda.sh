#!/usr/bin/env bash
# Build the serving image, push it to ECR, and create or update the Lambda
# function plus an API Gateway HTTP API in front of it.
#
# Requires: AWS CLI v2 configured (`aws configure`), Docker running,
#           models/model.joblib + models/metadata.json present.
#
#   ./scripts/deploy_lambda.sh
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
REPO="${ECR_REPO:-airbnb-price-api}"
FUNCTION="${LAMBDA_FUNCTION:-airbnb-price-api}"
ROLE_NAME="${LAMBDA_ROLE:-airbnb-price-lambda-role}"
API_NAME="${API_NAME:-airbnb-price-http-api}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPO}:latest"

[[ -f models/model.joblib && -f models/metadata.json ]] || {
  echo "No trained model in models/. Run training first."; exit 1; }

echo "==> ECR repository"
aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION" >/dev/null

echo "==> Build + push image"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
# linux/amd64 matches Lambda's default arch; --provenance=false avoids a manifest
# format Lambda rejects when building with buildx (Docker Desktop's default).
docker build --platform linux/amd64 --provenance=false -f Dockerfile.lambda -t "$IMAGE" .
docker push "$IMAGE"

echo "==> IAM role"
if ! ROLE_ARN="$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text 2>/dev/null)"; then
  ROLE_ARN="$(aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    --query Role.Arn --output text)"
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "    waiting for the new role to propagate..."; sleep 10
fi

echo "==> Lambda function"
if aws lambda get-function --function-name "$FUNCTION" --region "$REGION" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$FUNCTION" --image-uri "$IMAGE" \
    --region "$REGION" >/dev/null
else
  aws lambda create-function --function-name "$FUNCTION" --package-type Image \
    --code ImageUri="$IMAGE" --role "$ROLE_ARN" --memory-size 1024 --timeout 30 \
    --architectures x86_64 --region "$REGION" >/dev/null
fi
aws lambda wait function-active-v2 --function-name "$FUNCTION" --region "$REGION"
aws lambda wait function-updated-v2 --function-name "$FUNCTION" --region "$REGION"
FUNCTION_ARN="$(aws lambda get-function --function-name "$FUNCTION" --region "$REGION" \
  --query Configuration.FunctionArn --output text)"

echo "==> API Gateway (HTTP API)"
API_ID="$(aws apigatewayv2 get-apis --region "$REGION" \
  --query "Items[?Name=='${API_NAME}'].ApiId | [0]" --output text)"
if [[ "$API_ID" == "None" || -z "$API_ID" ]]; then
  # "Quick create": a $default route + auto-deployed $default stage pointing at the Lambda.
  API_ID="$(aws apigatewayv2 create-api --name "$API_NAME" --protocol-type HTTP \
    --target "$FUNCTION_ARN" --region "$REGION" --query ApiId --output text)"
  aws lambda add-permission --function-name "$FUNCTION" --statement-id apigw-invoke \
    --action lambda:InvokeFunction --principal apigateway.amazonaws.com \
    --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_ID}/*" \
    --region "$REGION" >/dev/null
fi
API_URL="https://${API_ID}.execute-api.${REGION}.amazonaws.com"

echo
echo "Deployed. API base URL: ${API_URL}"
echo "Test it:"
echo "  curl ${API_URL}/health"
echo "  curl -X POST ${API_URL}/predict -H 'Content-Type: application/json' -d @scripts/sample_request.json"
