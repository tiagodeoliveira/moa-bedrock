import asyncio
import boto3
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

bedrock = boto3.client('bedrock-runtime')

reference_models = [
    "ai21.j2-ultra-v1",
    "cohere.command-r-plus-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "mistral.mixtral-8x7b-instruct-v0:1",
]
aggregator_model = "anthropic.claude-3-sonnet-20240229-v1:0"
aggregator_prompt = """
You are a helpful assistant who aggregates and summarizes the responses from multiple AI models.
Please provide a comprehensive summary that combines the insights from all models but don't mention that this was a multi-model conversation.

Responses from models:
"""

def get_bedrock_response(model, prompt):
    response = bedrock.converse(
        modelId=model,
        inferenceConfig={'maxTokens': 1000, 'temperature': 0.7},
        messages=[{
            'role': 'user',
                    'content': [{'text': prompt}]
        }]
    )
    return response['output']['message']['content'][0]['text']

async def get_model_response(model, prompt):
    logger.info(f"Getting response from {model}")
    try:
        response = await asyncio.to_thread(get_bedrock_response, model, prompt)
        logger.info(f"Response from {model}: {response}")
        return response
    except Exception as e:
        logger.error(f"Error getting response from {model}: {str(e)}")
        return ""


async def mixture_of_agents(user_prompt):
    responses = await asyncio.gather(*[get_model_response(model, user_prompt)
             for model in reference_models])

    prompt = aggregator_prompt + "\n\n".join(responses)
    return await get_model_response(aggregator_model, prompt)

async def main():
    user_prompt = "Explain protein folding to a non-chemist."
    result = await mixture_of_agents(user_prompt)
    logger.info(f"Final response: \n {result}")

if __name__ == "__main__":
    asyncio.run(main())
