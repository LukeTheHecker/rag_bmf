import openai
from constants import OPENAI_API_KEY, OPENAI_MODEL_NAME, TEMPERATURE, MAX_TOKENS
from .prompts import SYSTEM_PROMPT

# Check if the OPENAI_API_KEY is set in the constants.py file
if OPENAI_API_KEY is None or len(OPENAI_API_KEY) == 0:
    raise ValueError("OPENAI_API_KEY is not set in the constants.py file")

class ChatGPTClient:
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.system_prompt = SYSTEM_PROMPT
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS

    def generate_response(self, prompt):
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

