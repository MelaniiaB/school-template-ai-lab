import telepot
from telepot.loop import MessageLoop
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

def message(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)

    print(content_type, chat_type, chat_id)

    completions = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
           {
           "role": "system",
           "content": "You are a helpful assistant."
           },
           {
           "role": "user",
           "content": msg['text']
           }
      ]
   )

    bot.sendMessage(chat_id, completions.choices[0].message.content)

bot = telepot.Bot("7950956474:AAHvx-rVL5rJkpD-zjKdeNHA8gMYVSFgGJE")
MessageLoop(bot, {'chat': message}).run_forever()

