import requests

def train_initial():
    # Change to your actual FastAPI base URL or port
    base_url = "http://127.0.0.1:5012"

    # 1) Sample initial training data for multiple agents
    # Each item has agent_type and a prompt
    initial_training_data = {
        "training_data": [
            {"agent_type": "general",   "prompt": "summarize what is being discussed in the document"},
            {"agent_type": "general",   "prompt": "tell me about the details of this topic in discussion"},
            {"agent_type": "general",   "prompt": "what are the main points of the document?"},

            {"agent_type": "legal",   "prompt": "summarize what is being discussed in the document"},
            {"agent_type": "legal",   "prompt": "highlight important legal clauses"},
            {"agent_type": "legal",   "prompt": "what are the compliance requirements?"},
            # MEETINGS agent
            {"agent_type": "meetings","prompt": "summarize what is being discussed in the meeting"},
            {"agent_type": "meetings","prompt": "summarize the latest meeting agenda"},
            {"agent_type": "meetings","prompt": "list all action items from the meeting"},
            {"agent_type": "meetings","prompt": "highlight the main decisions we took"},
            # FINANCE agent
            {"agent_type": "finance", "prompt": "please summarize the financial report"},
            {"agent_type": "finance", "prompt": "extract the key revenue metrics"},
            {"agent_type": "finance", "prompt": "what are the net profit details?"},

            {"agent_type": "tech",   "prompt": "summarize what is being discussed in the document"},
            {"agent_type": "tech",   "prompt": "tell me about the details of this topic in discussion"},
            {"agent_type": "tech",   "prompt": "what are the main points of the document?"}
        ]
    }

    # 2) Send a request to the /api/train-initial-data endpoint
    print("Sending initial training data...")
    response = requests.post(f"{base_url}/api/train-initial-data", json=initial_training_data)
    if response.status_code == 200:
        print("Initial training data processed successfully:", response.json())
    else:
        print("Error training data:", response.status_code, response.text)
        return

    # 3) Test a get-suggestions call for each agent to see if seeding worked
    # For example, we provide a partial input "summarize" to see suggestions
    test_queries = [
        {"agent_type": "legal", "partial_input": "summarize"},
        {"agent_type": "meetings", "partial_input": "summarize"},
        {"agent_type": "finance", "partial_input": "summarize"},
        {"agent_type": "general", "partial_input": "summarize"},
    ]

    for tq in test_queries:
        agent_type = tq["agent_type"]
        partial_input = tq["partial_input"]
        print(f"\nGetting suggestions for agent '{agent_type}' with partial input: '{partial_input}'")
        suggestions_payload = {
            "agent_type": agent_type,
            "input": partial_input,
            "max_suggestions": 3  # For brevity, let's ask for 3 suggestions
        }
        resp = requests.post(f"{base_url}/api/get-suggestions", json=suggestions_payload)
        if resp.status_code == 200:
            suggestions = resp.json().get("suggestions", [])
            for idx, s in enumerate(suggestions, start=1):
                print(f"  {idx}. {s['suggested_query']} (confidence={s['confidence']:.4f})")
        else:
            print(f"Error fetching suggestions for agent '{agent_type}':", resp.status_code, resp.text)

    # 4) Optionally save model state after training
    print("\nSaving model state (optional step)...")
    save_resp = requests.post(f"{base_url}/api/save-model-state")
    if save_resp.status_code == 200:
        print("Model state saved:", save_resp.json())
    else:
        print("Error saving model state:", save_resp.status_code, save_resp.text)
