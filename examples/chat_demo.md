# Streaming Chat Demo & Integration Guide - Open Source SDK

Welcome to the Open Source Blackbird SDK streaming chat tutorial! This guide will show you how to:
- Use the SDK for chat (regular and streaming)
- Embed streaming chat in your own Python applications
- Build a simple UI app that streams chat responses in real time
- **Correctly handle and clean streaming responses for your UI**

---

## 0. Start the Async Backend (Required for Streaming)

Before running any chat demo, you must start the backend in async (keepalive) mode. This ensures the backend stays running and streaming is fast and reliable.

**How to start the async backend:**

1. Create a file called `start_async_backend.py` with the following content:
   ```python
   from open_source_sdk import BlackbirdSDK

   if __name__ == "__main__":
       sdk = BlackbirdSDK(runasync=True)
       print("Async backend started. You can now run chat scripts in other terminals.")
       input("Press Enter to exit this script (the backend will keep running)...")
   ```
2. Run this script in a separate terminal:
   ```bash
   python start_async_backend.py
   ```
3. Leave this terminal open while you use the chat demo in another terminal or app.

**Why async mode?**
- The backend stays alive and is not restarted for every SDK session.
- Streaming responses are much faster and more reliable.
- You can run multiple chat scripts or UI apps without backend conflicts.

---

## 1. Prerequisites

- Python 3.8+
- Open Source Blackbird SDK installed
- Backend server running (see SDK docs for async backend setup)

---

## 2. Basic Chat Usage

### Regular (Non-Streaming) Chat
```python
from open_source_sdk import BlackbirdSDK

sdk = BlackbirdSDK()
chat_service = sdk.chat_service

response = chat_service.send_message(
    "Hello, how are you?",
    options={"agent": "general"}
)
print("Response:", response)
```

### Streaming Chat (Recommended for UI/Apps)
```python
from open_source_sdk import BlackbirdSDK
import json

sdk = BlackbirdSDK()
chat_service = sdk.chat_service

print("Streaming response:")
for chunk in chat_service.stream_chat_response(
    "Tell me a story about AI.",
    agent="general"
):
    # Each chunk is a line of data, usually starting with 'data: '
    if chunk.startswith('data: '):
        try:
            data = json.loads(chunk[6:])
            # Only print the actual bot message
            if 'response' in data and data['response']:
                print(data['response'], end='', flush=True)
        except Exception:
            pass
```

---

## 3. Handling and Cleaning Streaming Responses

**The streaming response from the backend is in Server-Sent Events (SSE) format, with each line like:**
```
data: {"response": "Hello! ", "tokens_per_second": 1.2}
data: {"response": "How can I help you?", "tokens_per_second": 1.3}
data: {"status": "complete"}
```

- **You should parse each chunk as JSON after removing the `data: ` prefix.**
- **Display only the `response` field to your users.**
- Ignore lines with only status or empty responses.

---

## 4. Handling the "replace" Field and Avoiding Duplicates

Sometimes, the backend will send a chunk with a `"replace": true` field. This means the previous bot response should be replaced with the new one (usually the final, clean answer). If you simply append every chunk, you may see duplicate or repeated answers in your UI.

**How to handle this:**
- Track where the bot's response starts in your UI.
- If a chunk with `"replace": true` arrives, clear the previous bot response and insert the new one.
- Otherwise, append as usual.

### Example: Clean Streaming in a UI with 'replace' Handling
```python
import json
...
def stream_response(self, message):
    self.append_text("Bot: ")
    bot_start_index = self.text_area.index(tk.END)  # Mark where the bot's response starts
    for chunk in self.chat_service.stream_chat_response(message, agent="general"):
        if chunk.startswith('data: '):
            try:
                data = json.loads(chunk[6:])
                if 'replace' in data and data['replace']:
                    # Remove the previous bot response and replace with this one
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
```

---

## 5. Embedding Streaming Chat in a Simple Python UI

We'll use Tkinter for a minimal, cross-platform GUI.

### File: `chat_streaming_ui.py`
```python
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
```

---

## 6. Running the Demo

1. Make sure your backend is running (async mode recommended for best performance).
2. Save the above code as `chat_streaming_ui.py` in your project.
3. Run:
   ```bash
   python chat_streaming_ui.py
   ```
4. Type a message and press Enter or click Send. Watch the bot's response stream in real time!

---

## 7. Troubleshooting

- **No response?** Make sure the backend is running and accessible at the expected port (default: 5012).
- **Import errors?** Ensure the SDK is installed and your Python environment is activated.
- **UI freezes?** The demo uses threads to keep the UI responsive. If you modify it, always stream in a background thread.
- **Streaming not working?** Check your backend version and that the `/chat` endpoint supports streaming (Flask/FastAPI with `yield` or SSE).

---

## 8. Next Steps & Customization

- Integrate this pattern into your own apps (web, desktop, etc.)
- Style the UI, add chat history, or support multiple agents
- Use the SDK's advanced features (file upload, RAG, agent creation)

---

## 9. Support & Feedback

- For SDK issues, open an issue on GitHub or contact the maintainers.
- For backend/server issues, check the backend logs and docs.

Happy building! 🚀 