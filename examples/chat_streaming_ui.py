import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from threading import Thread
from open_source_sdk import BlackbirdSDK
import json

class ChatUI:
    def __init__(self, root):
        self.sdk = BlackbirdSDK()
        self.chat_service = self.sdk.chat_service
        self.root = root
        self.root.title("Open Source Blackbird Streaming Chat Demo")

        self.text_area = ScrolledText(root, wrap=tk.WORD, height=20, width=60)
        self.text_area.pack(padx=10, pady=10)
        self.text_area.config(state=tk.DISABLED)

        self.entry = tk.Entry(root, width=50)
        self.entry.pack(side=tk.LEFT, padx=(10,0), pady=(0,10))
        self.entry.bind('<Return>', self.send_message)

        self.send_btn = tk.Button(root, text="Send", command=self.send_message)
        self.send_btn.pack(side=tk.LEFT, padx=10, pady=(0,10))

    def send_message(self, event=None):
        message = self.entry.get().strip()
        if not message:
            return
        self.entry.delete(0, tk.END)
        self.append_text(f"You: {message}\n")
        Thread(target=self.stream_response, args=(message,), daemon=True).start()

    def stream_response(self, message):
        self.append_text("Bot: ")
        bot_start_index = self.text_area.index(tk.END)
        for chunk in self.chat_service.stream_chat_response(message, agent="general"):
            if chunk.startswith('data: '):
                try:
                    data = json.loads(chunk[6:])
                    if 'replace' in data and data['replace']:
                        self.text_area.config(state=tk.NORMAL)
                        self.text_area.delete(bot_start_index, tk.END)
                        if 'response' in data and data['response']:
                            self.text_area.insert(tk.END, data['response'])
                        self.text_area.config(state=tk.DISABLED)
                    elif 'response' in data and data['response']:
                        self.append_text(data['response'])
                except Exception:
                    pass
        self.append_text("\n")

    def append_text(self, text):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, text)
        self.text_area.see(tk.END)
        self.text_area.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatUI(root)
    root.mainloop() 