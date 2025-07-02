import openai

# Replace this with your actual OpenAI API key
openai.api_key = "your-api-key"

# Define your blog topic
topic = "How AI is Transforming Small Businesses"

# Call the API
response = openai.ChatCompletion.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful blog-writing assistant."},
        {"role": "user", "content": f"Write a 500-word blog post on the topic: {topic}"}
    ],
    temperature=0.7
)

# Print the blog post
print(response['choices'][0]['message']['content'])
