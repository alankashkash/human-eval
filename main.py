import os

from human_eval.data import write_jsonl, read_problems
from langchain_openai import ChatOpenAI
import random


def generate_one_completion(input_sample):
    # IMPORTANT
    # replace this function with an invocation of your model
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=os.environ.get("OPENAI_API_KEY"), max_tokens=512)
    response = llm.invoke(input_sample).content
    return response

problems = read_problems()
num_samples_per_task = 2  # Reduced number of samples per task

samples = []
total_tasks = len(problems)

for task_id, (key, prompt_data) in enumerate(problems.items()):
    for i in range(num_samples_per_task):
        print(f"Processing task {task_id + 1}/{total_tasks}, sample {i + 1}/{num_samples_per_task}")
        completion = generate_one_completion(prompt_data["prompt"])
        samples.append(dict(task_id=key, completion=completion))

write_jsonl("samples.jsonl", samples)
