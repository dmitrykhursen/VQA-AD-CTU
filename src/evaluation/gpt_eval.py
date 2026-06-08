import os
import numpy as np
import json
from multiprocessing import Pool
from openai import OpenAI


class GPTEvaluation:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)

    def call_chatgpt(self, chatgpt_messages, max_tokens=40, model="gpt-3.5-turbo"):
        response = self.client.chat.completions.create(
            model=model, messages=chatgpt_messages, temperature=0.6, max_tokens=max_tokens
        )
        reply = response.choices[0].message.content
        total_tokens = response.usage.total_tokens
        return reply, total_tokens

    def prepare_chatgpt_message(self, prompt):
        system_message = "an evaluator who rates my answer based on the correct answer"
        messages = [{"role": "system", "content": system_message}]
        messages.append({"role": "user", "content": "{}".format(prompt)})
        return messages

    def forward(self, data):
        answer, GT = data
        prompts = "Rate my answer based on the correct answer out of 100, with higher scores indicating that the answer \
        is closer to the correct answer, and you should be accurate to single digits like 62, 78, 41,etc. Output the number only"
        prompts = prompts + "This is the correct answer: " + GT + "This is my answer: " + answer

        output = ""
        messages = self.prepare_chatgpt_message(prompts)
        reply, total_tokens = self.call_chatgpt(messages, max_tokens=300)

        output += reply
        output += "\n\n"

        output = output[:-2]

        return output


if __name__ == "__main__":
    data = [
        ("The ego vehicle should notice the bus next, as it is the third object in the image.", "Firstly, notice <c3,CAM_FRONT_LEFT,1075.5,382.8>."),
    ]

    eval = GPTEvaluation()

    with Pool(5) as p:
        scores = p.map(eval.forward, data)

    print(scores)
