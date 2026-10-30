from pydantic import BaseModel, ValidationError

TASKS, EVENTS = {}, {}

class CreateTaskArgs(BaseModel):
    owner: str
    task: str
    due: str | None = None

class ScheduleEventArgs(BaseModel):
    title: str
    participants: list[str]
    when: str

def create_task(args: CreateTaskArgs):
    task_id = f"TSK-{len(TASKS) + 1:04d}"
    TASKS[task_id] = args.model_dump()
    return {"task_id": task_id, "status": "created", **args.model_dump()}

def schedule_event(args: ScheduleEventArgs):
    event_id = f"EV-{len(EVENTS) + 1:04d}"
    EVENTS[event_id] = args.model_dump()
    return {"event_id": event_id, "status": "scheduled", **args.model_dump()}

TOOL_SPECS = {
    "create_task": {"description": "Create a task with an owner, description, and optional due date.", "args_model": CreateTaskArgs},
    "schedule_event": {"description": "Schedule an event with a title, participants, and a date/time.", "args_model": ScheduleEventArgs},
}

TOOLS = {
    "create_task": create_task,
    "schedule_event": schedule_event
}   

def run_tool(tool_name: str, args_dict: dict):
    if tool_name not in TOOLS:
        raise ValueError(f"Tool '{tool_name}' is not defined.")
    tool_func = TOOLS[tool_name]
    args_model = TOOL_SPECS[tool_name]["args_model"]
    try:
        args = args_model.model_validate(args_dict)
    except ValidationError as e:
        return {"error": f"Invalid arguments for tool '{tool_name}': {e}"}
    return tool_func(args)