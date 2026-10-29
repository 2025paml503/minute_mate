SYSTEM_ROLE = ("You are a careful minute taker for a small team. You write for a manager who did not read the chat. "
               "Write a concise summary of the conversation, highlighting key decisions, action items, and important points. "
               "Avoid including trivial details or off-topic discussions. Use clear and professional language. ")

def zero_shot(dialogue):
    return f"Summarize the following conversation:\n{dialogue}\n\nSummary:"

def few_shot(dialogue, examples):
    example_text = "\n\n".join([f"Conversation:\n{ex['dialogue']}\nSummary:\n{ex['summary']}" for ex in examples])
    return f"{example_text}\n\nConversation:\n{dialogue}\nSummary:"

def role_based(dialogue):
    return f"Write the minutes of this conversation as a short summary. Summarize the following conversation:\n{dialogue}\n\nSummary:"

def constrained(dialogue):
    return f"Write a concise summary of the following conversation in at most 50 words, highlighting key decisions, action items, and important points. Avoid including trivial details or off-topic discussions. Use clear and professional language.\n\nConversation:\n{dialogue}\n\nSummary:"

def detailed(dialogue):
    return f"Write a detailed summary of the following conversation, including key decisions, action items, and important points. Avoid including trivial details or off-topic discussions. Use clear and professional language.\n\nConversation:\n{dialogue}\n\nSummary:"