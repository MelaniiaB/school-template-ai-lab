import collections
import collections.abc
collections.Hashable = collections.abc.Hashable

import telepot
from telepot.loop import MessageLoop
from telepot.delegate import pave_event_space, per_chat_id, create_open
from openai import OpenAI
import os
import glob
from dotenv import load_dotenv
from mood_model import predict

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

class SafeMoodBot(telepot.helper.ChatHandler):
    def __init__(self, *args, **kwargs):
        super(SafeMoodBot, self).__init__(*args, **kwargs)

        self.history = []

    def on_chat_message(self, msg):
        print(msg)
        content_type, chat_type, chat_id = telepot.glance(msg)

        print(content_type, chat_type, chat_id)

        if content_type != 'video_note':
            return

        self.bot.download_file(msg['video_note']['file_id'], msg['video_note']['file_id'] + ".mp4")
        
        # extract audio from mp4 (wav format - no special encoder needed)
        os.system(f"ffmpeg -i {msg['video_note']['file_id']}.mp4 -vn {msg['video_note']['file_id']}.wav")
        # extract jpg frames from mp4
        os.system(f"ffmpeg -i {msg['video_note']['file_id']}.mp4 -vf fps=1 {msg['video_note']['file_id']}_%04d.jpg")


        audio_file = open(f"{msg['video_note']['file_id']}.wav", "rb")

        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe", 
            file=audio_file
        )

        mood = predict(f"{msg['video_note']['file_id']}_0001.jpg")
        #mood = "веселий"  # default mood

        text = transcription.text + ". У мене зараз настрій: " + mood
        print(text)
        self.history.append(text)

        completions = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "ти бот-психолог, допоможи людині. не реагуй на настрій напряму, але коригуй свої відповіді в залежності від нього. якщо людина у поганому настрої, то спробуй її підбадьорити, якщо про хороший - підтримай її.",
                },
                {
                    "role": "user",
                    "content": ". ".join(self.history)
                }
            ]
        )

        os.remove(msg['video_note']['file_id'] + ".mp4")
        os.remove(msg['video_note']['file_id'] + ".wav")
        # Remove extracted frames
        for f in glob.glob(msg['video_note']['file_id'] + "_*.jpg"):
            os.remove(f)

        self.sender.sendMessage(completions.choices[0].message.content)

if __name__ == "__main__":
    print("Starting bot...")
    bot = telepot.DelegatorBot(os.environ.get("BOT_KEY"), [
        pave_event_space()(
            per_chat_id(), create_open, SafeMoodBot, timeout=10),
    ])
    bot.deleteWebhook()
    MessageLoop(bot).run_forever()
