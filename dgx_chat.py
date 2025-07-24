import openai
with open('.nemotron_token', 'r') as token:
        TOKEN = token.readline().strip()

# Configure OpenAI client
client = openai.OpenAI(
    api_key= f"{TOKEN}",
    base_url= "https://obsidian.ccs.ornl.gov/ai/nemotron/api/v1"
)

print("Welcome to the Nemotron-8B chatbot! Type 'exit' to quit.")



while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        break
    print(user_input)
    try:
        chat_response = client.chat.completions.create(
            model="Llama-3_1-Nemotron-Ultra-253B-v1-FP8", 
            messages=[{"role": "user", "content": user_input}],
            temperature=0.7,
            top_p=1,
            max_tokens=512,
        )
        print("Bot:", chat_response.choices[0].message.content)
    except Exception as e:
        print("Error:", e)

# import openai
# with open('.nemotron_token', 'r') as token:
#         TOKEN = token.readline().strip()
 
# client = openai.OpenAI(
#         api_key = f"{TOKEN}",
#         base_url = "https://obsidian.ccs.ornl.gov/ai/nemotron/api/v1",
#     )
 
# chat_completion = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user", "content": "tell me about nemotron"
#                 }
#             ],
#         model="Llama-3_1-Nemotron-Ultra-253B-v1-FP8",
#         temperature=0.7,
#         top_p=1,
#         max_tokens=512,
# )
# print(chat_completion.choices[0].message.content)