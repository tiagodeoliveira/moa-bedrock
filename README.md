# Mixture of Agents

A Python script that uses multiple AI models via AWS Bedrock to generate and aggregate responses.

## Quick Start

1. Install dependencies:
   ```
   pip install boto3
   ```

2. Set up AWS credentials for Bedrock access.

3. Run the script:
   ```
   python script_name.py
   ```

## What it does

1. Sends a prompt to multiple AI models using AWS Bedrock.
2. Collects responses asynchronously.
3. Uses an aggregator model to combine responses into a summary.
4. Logs the final result.

## Configuration

- `reference_models`: List of model IDs to query.
- `aggregator_model`: Model ID for summarizing responses.
- `user_prompt`: The question to ask (in `main()` function).

## Requirements

- Python 3.7+
- boto3
- AWS account with Bedrock access

## Notes

- Logging is set to INFO level.
- Basic error handling is implemented for each model response.