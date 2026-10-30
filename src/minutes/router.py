from .tools import run_tool
MEETING_WORDS = ["meeting", "minutes", "agenda", "discussion", "action items"]

def propose_calls(minutes):
    """
    Propose calls to tools based on the content of the meeting minutes.
    Returns a list of dicts with 'tool_name' and 'args'.
    """
    calls = []
    # Check for action items to create tasks
    for item in minutes.action_items:
        if item.task:
            args = {
                "owner": item.owner or "unassigned",
                "task": item.task,
                "due": item.due
            }
            calls.append({"tool_name": "create_task", "args": args})
    
    # Check for open questions to schedule events
    if minutes.open_questions:
        args = {
            "title": "Follow-up Meeting",
            "participants": minutes.participants,
            "when": "TBD"  # Placeholder, could be improved with date parsing
        }
        calls.append({"tool_name": "schedule_event", "args": args})
    
    return calls

def execute_proposed_calls(calls):
    """
    Execute the proposed tool calls and return their results.
    """
    results = []
    for call in calls:
        tool_name = call["tool_name"]
        args = call["args"]
        result = run_tool(tool_name, args)
        results.append({"tool_name": tool_name, "result": result})
    return results