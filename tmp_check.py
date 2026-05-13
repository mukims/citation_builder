import ollama

response = ollama.chat(
    model="gemma4:latest",
    messages=[{"role": "user", "content": "hi"}],
    stream=False
)

print("Attributes:", [attr for attr in dir(response) if not attr.startswith('_')])
print("Prompt tokens:", getattr(response, 'prompt_eval_count', None))
print("Completion tokens:", getattr(response, 'eval_count', None))
